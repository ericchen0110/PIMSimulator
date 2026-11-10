#include "SparsePIM.h"
#include "MultiChannelMemorySystem.h"
#include "Utils.h"
#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

using namespace DRAMSim;

uint64_t SparsePIM::bankAddress(const SparsePIMPartition& p, uint64_t offset)
{
    if (PIMConfiguration::getAddressMappingScheme() != Scheme8)
        throw std::invalid_argument("Sparse PIM default placement requires Scheme8");
    const unsigned burst = getConfigParam(UINT, "BL") * getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") / 8;
    const unsigned cols = getConfigParam(UINT, "NUM_COLS") / getConfigParam(UINT, "BL");
    const unsigned banks = getConfigParam(UINT, "NUM_BANKS");
    const unsigned chans = getConfigParam(UINT, "NUM_CHANS");
    const unsigned rows = getConfigParam(UINT, "NUM_ROWS");
    const uint64_t slot = offset / burst;
    const uint64_t row = slot / (2 * cols);
    if (row >= std::min(rows, 1U << 13) || p.channel >= chans ||
        p.rank >= getConfigParam(UINT, "NUM_RANKS") ||
        p.block >= getConfigParam(UINT, "NUM_PIM_BLOCKS") || 2 * p.block + 1 >= banks)
        throw std::out_of_range("Sparse PIM placement exceeds bank pair capacity");
    uint64_t addr = (uint64_t(p.rank) << uLog2(rows)) | row;
    addr = (addr << uLog2(cols)) | ((slot / 2) % cols);
    addr = (addr << uLog2(banks)) | (2 * p.block + slot % 2);
    addr = (addr << uLog2(chans)) | p.channel;
    return addr << uLog2(burst);
}

SparsePIM::SparsePIM(MultiChannelMemorySystem& mem, const SparsePIMJob& descriptor,
                     SparsePIMPolicy mode, unsigned latency)
    : memory(mem), job(descriptor), policy(mode), macLatency(latency),
      blocks_(job.partitions.size())
{
#ifdef NO_STORAGE
    throw std::logic_error("Sparse PIM requires a data-enabled build (NO_STORAGE=0)");
#endif
    if (!macLatency || !job.columns) throw std::invalid_argument("Invalid sparse PIM dimensions/latency");
    if (getConfigParam(UINT, "NUM_BANKS") != 2 * getConfigParam(UINT, "NUM_PIM_BLOCKS") ||
        getConfigParam(UINT, "BL") * getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") / 8 != sizeof(BurstType))
        throw std::invalid_argument("Sparse PIM requires bank pairs and 32-byte bursts");
    if (!job.address) job.address = bankAddress;
    std::set<std::tuple<unsigned, unsigned, unsigned>> owners;
    std::map<std::pair<unsigned, unsigned>, std::vector<size_t>> rankGroups;
    for (size_t i = 0; i < job.partitions.size(); ++i)
    {
        const auto& p = job.partitions[i];
        bankAddress(p, 0); // Validate ownership even with a placement adapter.
        if (!owners.emplace(p.channel, p.rank, p.block).second)
            throw std::invalid_argument("Duplicate sparse PIM block owner");
        for (uint64_t offset : {p.rowPtr, p.columns, p.values, p.input, p.output})
            if (offset % sizeof(BurstType)) throw std::invalid_argument("Sparse PIM arrays must be burst aligned");
        rankGroups[{p.channel, p.rank}].push_back(i);
        if (!p.rows) blocks_[i].pc = PC::Done;
    }
    for (const auto& entry : rankGroups) groups.push_back(entry.second);
}

bool SparsePIM::busy() const
{
    for (const auto& b : blocks_) if (b.pc != PC::Done || b.waiting) return true;
    return false;
}

void SparsePIM::request(size_t id, uint64_t offset, bool write, PC next)
{
    auto& b = blocks_[id];
    const uint64_t addr = job.address(job.partitions[id], offset);
    if (!memory.willAcceptTransaction(addr)) return; // Retry without advancing the PC.
    const unsigned lane = (offset % sizeof(BurstType)) / sizeof(uint32_t);
    auto* transaction = new Transaction(write ? DATA_WRITE : DATA_READ, addr,
                                        "SPARSE_PIM", &b.buffer);
    transaction->onComplete = [this, id, lane, next, write]() {
        auto& state = blocks_[id];
        if (!write)
        {
            switch (state.pc)
            {
                case PC::RowStart: state.cursor = state.buffer.u32Data_[lane]; break;
                case PC::RowEnd:
                    state.end = state.buffer.u32Data_[lane];
                    if (state.end < state.cursor || state.end > job.partitions[id].entries)
                        throw std::runtime_error("Invalid CSR row pointer");
                    break;
                case PC::Column:
                    state.column = state.buffer.u32Data_[lane];
                    if (state.column >= job.columns) throw std::runtime_error("CSR column out of range");
                    break;
                case PC::Value: state.value = state.buffer.fp32Data_[lane]; break;
                case PC::Input: state.input = state.buffer.fp32Data_[lane]; break;
                default: throw std::logic_error("Unexpected sparse PIM read completion");
            }
        }
        state.waiting = false;
        state.pc = next;
        if (next == PC::Mac) state.readyCycle = cycle + macLatency;
        if (next == PC::Column && state.cursor == state.end) state.pc = PC::RowBarrier;
    };
    b.waiting = true;
    if (!memory.addTransaction(transaction))
    {
        b.waiting = false;
        delete transaction;
    }
    else
        ++stats_.instructions;
}

void SparsePIM::advance(size_t id)
{
    auto& b = blocks_[id];
    const auto& p = job.partitions[id];
    if (b.waiting) { ++stats_.memoryWaitCycles; return; }
    switch (b.pc)
    {
        case PC::RowStart: request(id, p.rowPtr + 4ULL * b.row, false, PC::RowEnd); break;
        case PC::RowEnd: request(id, p.rowPtr + 4ULL * (b.row + 1), false, PC::Column); break;
        case PC::Column: request(id, p.columns + 4ULL * b.cursor, false, PC::Value); break;
        case PC::Value: request(id, p.values + 4ULL * b.cursor, false, PC::Input); break;
        case PC::Input: request(id, p.input + 4ULL * b.column, false, PC::Mac); break;
        case PC::Mac:
            if (cycle < b.readyCycle) break;
            // Scalar FP32 sparse datapath: one nonzero per issue, no reassociation.
            b.accumulator = b.accumulator + b.value * b.input;
            ++stats_.instructions;
            ++b.macs; ++stats_.macs; ++b.cursor;
            b.pc = b.cursor == b.end ? PC::RowBarrier : PC::EntryBarrier;
            break;
        case PC::EntryBarrier: case PC::RowBarrier: ++stats_.barrierCycles; break;
        case PC::Store:
            // One result per burst avoids read-modify-write races between rows.
            b.buffer = BurstType();
            b.buffer.fp32Data_[0] = b.accumulator;
            request(id, p.output + sizeof(BurstType) * b.row, true, PC::AdvanceRow);
            break;
        case PC::AdvanceRow:
            ++b.row; ++b.completedRows; ++stats_.completedRows;
            b.accumulator = 0;
            b.pc = b.row == p.rows ? PC::Done : PC::RowStart;
            break;
        case PC::Done: break;
    }
}

void SparsePIM::releaseBarriers()
{
    // Rank-local sparse-slot barriers. All decisions precede PC updates.
    for (const auto& group : groups)
    {
        bool slotReady = true, rowReady = true;
        for (auto id : group)
        {
            const auto pc = blocks_[id].pc;
            slotReady &= pc == PC::EntryBarrier || pc == PC::RowBarrier || pc == PC::Done;
            rowReady &= pc == PC::RowBarrier || pc == PC::Done;
        }
        for (auto id : group)
        {
            auto& b = blocks_[id];
            if (b.pc == PC::EntryBarrier && (policy == SparsePIMPolicy::Independent || slotReady))
                b.pc = PC::Column;
            else if (b.pc == PC::RowBarrier && (policy == SparsePIMPolicy::Independent || rowReady))
                b.pc = PC::Store;
        }
    }
}

void SparsePIM::update()
{
    if (!busy()) return;
    releaseBarriers();
    for (size_t i = 0; i < blocks_.size(); ++i) advance(i);
    for (const auto& group : groups)
    {
        uint64_t low = UINT64_MAX, high = 0;
        for (auto id : group)
        {
            low = std::min(low, uint64_t(blocks_[id].row));
            high = std::max(high, uint64_t(blocks_[id].row));
        }
        stats_.maxRowLead = std::max(stats_.maxRowLead, high - low);
    }
    stats_.cycles = ++cycle;
}

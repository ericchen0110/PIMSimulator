#include "SparseExecution.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "SystemConfiguration.h"

namespace DRAMSim
{

namespace
{
const uint32_t kPimRegRow = 0x3fff;
const uint32_t kPimAbmrRow = 0x27ff;
const uint32_t kPimSbmrRow = 0x2fff;
const uint32_t kPimParkRow = 1u << 13;

uint32_t ceilDiv(uint32_t value, uint32_t divisor)
{
    return value == 0 ? 0 : (value + divisor - 1) / divisor;
}
}  // namespace

const char* requestClassName(RequestClass requestClass)
{
    switch (requestClass)
    {
        case RequestClass::MATRIX_VALUE:
            return "matrix-value";
        case RequestClass::MATRIX_INDEX:
            return "matrix-index";
        case RequestClass::DENSE_VECTOR:
            return "dense-vector";
        case RequestClass::RESULT:
            return "result";
        case RequestClass::PIM_CONTROL:
            return "pim-control";
        case RequestClass::OTHER:
            return "other";
    }
    return "unknown";
}

const char* sparsePhaseName(SparsePhase phase)
{
    switch (phase)
    {
        case SparsePhase::MATRIX_SETUP:
            return "matrix-setup";
        case SparsePhase::VECTOR_STAGING:
            return "vector-staging";
        case SparsePhase::PIM_COMPUTE:
            return "pim-compute";
        case SparsePhase::RESULT_READBACK:
            return "result-readback";
    }
    return "unknown";
}

SparseTrafficCounter::SparseTrafficCounter()
    : transactions(0), reads(0), writes(0), bytes(0), effectiveBankBursts(0)
{
}

SparsePhaseCycles::SparsePhaseCycles()
    : matrixSetup(0),
      vectorStaging(0),
      pimCompute(0),
      resultReadback(0),
      recurringSpmv(0),
      endToEnd(0)
{
}

SparseRunStats::SparseRunStats()
    : usefulSegments(0),
      scheduledSegments(0),
      wavePaddingSegments(0),
      usefulVectorBursts(0),
      paddedVectorBursts(0),
      grfOverwriteBursts(0),
      reloadRounds(0),
      cpuIndexBytes(0),
      cpuGatherValueBytes(0)
{
}

HostGatherLockstepExecutor::HostGatherLockstepExecutor(
    std::shared_ptr<MultiChannelMemorySystem> memory, uint32_t pimChannels, uint32_t pimRanks)
    : memory_(memory), channels_(pimChannels), ranks_(pimRanks)
{
    if (!memory_ || channels_ == 0 || ranks_ == 0)
        throw std::invalid_argument("host-gather executor needs memory, channels, and ranks");
    banks_ = getConfigParam(UINT, "NUM_BANKS");
    bankGroups_ = getConfigParam(UINT, "NUM_BANK_GROUPS");
    pimBlocks_ = getConfigParam(UINT, "NUM_PIM_BLOCKS");
    uint32_t rows = getConfigParam(UINT, "NUM_ROWS");
    accessibleRows_ = std::min<uint32_t>(rows, 1u << 13);
    columnsPerBurstRow_ = getConfigParam(UINT, "NUM_COLS") / getConfigParam(UINT, "BL");
    if (banks_ != pimBlocks_ * 2)
        throw std::invalid_argument("host-gather baseline requires two banks per PIM block");
    if (channels_ > getConfigParam(UINT, "NUM_CHANS"))
        throw std::invalid_argument("requested PIM channels exceed configured memory channels");

    addressManager_ = std::make_shared<PIMAddrManager>(channels_, ranks_);
    controlOn_.u8Data_[0] = 1;
    controlOn_.u8Data_[16] = 0;
    controlOn_.u8Data_[20] = 0;
    controlOn_.u8Data_[21] = 1;
    controlOff_.u8Data_[0] = 0;
    controlOff_.u8Data_[16] = 0;
}

SparseExecutionLayout HostGatherLockstepExecutor::buildLayout(
    const SparseMatrixCSR& matrix, const SparseMappingPlan& plan) const
{
    matrix.validate();
    if (plan.numRows != matrix.numRows || plan.numCols != matrix.numCols)
        throw std::invalid_argument("sparse execution plan dimensions do not match the matrix");
    if (plan.workers.size() != static_cast<size_t>(channels_) * ranks_ * pimBlocks_)
        throw std::invalid_argument("sparse mapping worker count does not match PIM topology");

    SparseExecutionLayout layout;
    layout.resultBursts.resize(matrix.numRows);
    uint64_t storageSize = 0;

    for (uint32_t channel = 0; channel < channels_; ++channel)
    {
        for (uint32_t rank = 0; rank < ranks_; ++rank)
        {
            size_t baseWorker = (static_cast<size_t>(channel) * ranks_ + rank) * pimBlocks_;
            size_t waveCount = 0;
            for (uint32_t block = 0; block < pimBlocks_; ++block)
                waveCount = std::max(waveCount, plan.workers[baseWorker + block].rowIds.size());

            uint32_t dataRowCursor = 0;
            size_t firstWave = layout.waves.size();
            for (size_t waveIndex = 0; waveIndex < waveCount; ++waveIndex)
            {
                SparseLockstepWave wave;
                wave.channel = channel;
                wave.rank = rank;
                wave.waveIndex = static_cast<uint32_t>(waveIndex);
                wave.segmentCount = 0;
                wave.dataBaseRow = dataRowCursor;
                wave.resultRow = 0;
                wave.storageOffset = storageSize;
                wave.rowByBlock.assign(pimBlocks_, -1);

                for (uint32_t block = 0; block < pimBlocks_; ++block)
                {
                    const SparseWorkerQueue& worker = plan.workers[baseWorker + block];
                    if (waveIndex >= worker.rowIds.size())
                        continue;
                    uint32_t row = worker.rowIds[waveIndex];
                    wave.rowByBlock[block] = row;
                    wave.segmentCount = std::max(wave.segmentCount, matrix.rowSegments(row));
                }
                if (wave.segmentCount == 0)
                    continue;

                // FILL and auto-indexed MAC advance through all eight GRF entries in the current
                // ISA, so a lockstep wave must issue whole eight-burst register tiles.
                wave.segmentCount = ceilDiv(wave.segmentCount, 8) * 8;

                uint32_t dataRows = 2 * ceilDiv(wave.segmentCount, 8);
                dataRowCursor += dataRows;
                storageSize += static_cast<uint64_t>(pimBlocks_) * wave.segmentCount;
                layout.waves.push_back(wave);
            }

            uint32_t resultRow = dataRowCursor;
            for (size_t waveIndex = firstWave; waveIndex < layout.waves.size(); ++waveIndex)
            {
                SparseLockstepWave& wave = layout.waves[waveIndex];
                if (wave.channel != channel || wave.rank != rank)
                    break;
                wave.resultRow = resultRow++;
            }
            if (resultRow > accessibleRows_)
                throw std::runtime_error("sparse layout exceeds accessible PIM DRAM rows");
        }
    }

    layout.weightBursts.resize(storageSize);
    layout.vectorBursts.resize(storageSize);
    layout.crfBursts.resize(layout.waves.size());
    for (const SparseLockstepWave& wave : layout.waves)
    {
        for (uint32_t block = 0; block < pimBlocks_; ++block)
        {
            int64_t row = wave.rowByBlock[block];
            if (row < 0)
                continue;
            uint32_t segmentCount = matrix.rowSegments(static_cast<uint32_t>(row));
            for (uint32_t segment = 0; segment < segmentCount; ++segment)
            {
                uint64_t index = wave.storageOffset +
                                 static_cast<uint64_t>(block) * wave.segmentCount + segment;
                layout.weightBursts[index] =
                    plan.segment(static_cast<uint32_t>(row), segment).values;
            }
        }
    }
    return layout;
}

uint64_t HostGatherLockstepExecutor::address(uint32_t channel, uint32_t rank, uint32_t flatBank,
                                             uint32_t row, uint32_t column) const
{
    if (flatBank >= banks_ || row >= getConfigParam(UINT, "NUM_ROWS") ||
        column >= columnsPerBurstRow_)
        throw std::out_of_range("sparse physical address component is outside configuration");
    uint32_t banksPerGroup = banks_ / bankGroups_;
    return addressManager_->addrGen(channel, rank, flatBank / banksPerGroup,
                                    flatBank % banksPerGroup, row, column);
}

uint64_t HostGatherLockstepExecutor::pimAddress(uint32_t channel, uint32_t rank,
                                                uint32_t bankParity, uint32_t row,
                                                uint32_t column) const
{
    return addressManager_->addrGen(channel, rank, 0, bankParity, row, column);
}

void HostGatherLockstepExecutor::submit(bool isWrite, uint64_t transactionAddress,
                                        const std::string& tag, BurstType* data,
                                        RequestClass requestClass, SparsePhase phase,
                                        SparseRunStats& stats, uint32_t effectiveBankBursts)
{
    TransactionType type = isWrite ? DATA_WRITE : DATA_READ;
    Transaction* transaction =
        new Transaction(type, transactionAddress, tag, data, requestClass);
    if (!memory_->addTransaction(transaction))
    {
        delete transaction;
        throw std::runtime_error("memory system rejected sparse transaction");
    }

    SparseTrafficCounter& traffic = stats.traffic[phase];
    traffic.transactions++;
    traffic.bytes += 32;
    traffic.effectiveBankBursts += effectiveBankBursts;
    SparseTrafficCounter::ByClass& classTraffic = traffic.byClass[requestClass];
    classTraffic.transactions++;
    classTraffic.bytes += 32;
    classTraffic.effectiveBankBursts += effectiveBankBursts;
    if (isWrite)
    {
        traffic.writes++;
        classTraffic.writes++;
    }
    else
    {
        traffic.reads++;
        classTraffic.reads++;
    }
}

void HostGatherLockstepExecutor::barrier(uint32_t channel)
{
    memory_->addBarrier(channel);
}

uint64_t HostGatherLockstepExecutor::drain()
{
    uint64_t start = memory_->currentClockCycle;
    while (memory_->hasPendingTransactions()) memory_->update();
    return memory_->currentClockCycle - start;
}

void HostGatherLockstepExecutor::queueMatrixSetup(const SparseMatrixCSR&,
                                                  const SparseMappingPlan&,
                                                  SparseExecutionLayout& layout,
                                                  SparseRunStats& stats)
{
    for (const SparseLockstepWave& wave : layout.waves)
    {
        for (uint32_t block = 0; block < pimBlocks_; ++block)
        {
            for (uint32_t segment = 0; segment < wave.segmentCount; ++segment)
            {
                uint64_t storage = wave.storageOffset +
                                   static_cast<uint64_t>(block) * wave.segmentCount + segment;
                uint32_t row = wave.dataBaseRow + 2 * (segment / 8);
                uint32_t column = segment % 8;
                submit(true, address(wave.channel, wave.rank, block * 2 + 1, row, column),
                       "SPARSE_WEIGHT", &layout.weightBursts[storage], RequestClass::MATRIX_VALUE,
                       SparsePhase::MATRIX_SETUP, stats);
            }
        }
    }
}

void HostGatherLockstepExecutor::queueVectorStaging(const SparseMappingPlan& plan,
                                                    const std::vector<fp16>& input,
                                                    SparseExecutionLayout& layout,
                                                    SparseRunStats& stats)
{
    if (input.size() != plan.numCols)
        throw std::invalid_argument("SpMV input vector length must equal matrix columns");

    for (const SparseLockstepWave& wave : layout.waves)
    {
        for (uint32_t block = 0; block < pimBlocks_; ++block)
        {
            int64_t rowId = wave.rowByBlock[block];
            uint32_t usefulSegments =
                rowId < 0 ? 0 : plan.rowSegmentOffsets[rowId + 1] - plan.rowSegmentOffsets[rowId];
            for (uint32_t segment = 0; segment < wave.segmentCount; ++segment)
            {
                uint64_t storage = wave.storageOffset +
                                   static_cast<uint64_t>(block) * wave.segmentCount + segment;
                BurstType& gathered = layout.vectorBursts[storage];
                gathered = BurstType();
                if (segment < usefulSegments)
                {
                    const SparseSegment& sparseSegment =
                        plan.segment(static_cast<uint32_t>(rowId), segment);
                    for (uint32_t lane = 0; lane < 16; ++lane)
                    {
                        if (sparseSegment.validLaneMask & (1u << lane))
                            gathered.fp16Data_[lane] = input[sparseSegment.columnIndices[lane]];
                    }
                }
                uint32_t row = wave.dataBaseRow + 2 * (segment / 8);
                uint32_t column = segment % 8;
                submit(true, address(wave.channel, wave.rank, block * 2, row, column),
                       "SPARSE_VECTOR", &gathered, RequestClass::DENSE_VECTOR,
                       SparsePhase::VECTOR_STAGING, stats);
            }
        }
    }
}

void HostGatherLockstepExecutor::parkIn(uint32_t channel, uint32_t rank, SparseRunStats& stats)
{
    for (uint32_t bank = 0; bank < banks_; ++bank)
        submit(false, address(channel, rank, bank, kPimParkRow, 0), "SPARSE_PARK_IN", &nullBurst_,
               RequestClass::PIM_CONTROL, SparsePhase::PIM_COMPUTE, stats);
    barrier(channel);
}

void HostGatherLockstepExecutor::parkOut(uint32_t channel, uint32_t rank, SparseRunStats& stats)
{
    for (uint32_t bank = 0; bank < banks_; ++bank)
        submit(false, address(channel, rank, bank, kPimParkRow, 0), "SPARSE_PARK_OUT", &nullBurst_,
               RequestClass::PIM_CONTROL, SparsePhase::PIM_COMPUTE, stats);
    barrier(channel);
}

void HostGatherLockstepExecutor::enterHab(uint32_t channel, uint32_t rank, SparseRunStats& stats)
{
    const uint32_t modeBanks[] = {0, 1, 8, 9};
    for (uint32_t bank : modeBanks)
        submit(true, address(channel, rank, bank, kPimAbmrRow, 0x1f), "SPARSE_SB_TO_HAB",
               &nullBurst_, RequestClass::PIM_CONTROL, SparsePhase::PIM_COMPUTE, stats);
    barrier(channel);
}

void HostGatherLockstepExecutor::exitHab(uint32_t channel, uint32_t rank, SparseRunStats& stats)
{
    submit(true, pimAddress(channel, rank, 0, kPimSbmrRow, 0x1f), "SPARSE_HAB_TO_SB",
           &nullBurst_, RequestClass::PIM_CONTROL, SparsePhase::PIM_COMPUTE, stats);
    submit(true, pimAddress(channel, rank, 1, kPimSbmrRow, 0x1f), "SPARSE_HAB_TO_SB",
           &nullBurst_, RequestClass::PIM_CONTROL, SparsePhase::PIM_COMPUTE, stats);
    barrier(channel);
}

void HostGatherLockstepExecutor::programWave(uint32_t channel, uint32_t rank,
                                             SparseLockstepWave& wave, BurstType& crfBurst,
                                             SparseRunStats& stats)
{
    PIMCmd nop(PIMCmdType::NOP, 0);
    for (uint32_t index = 0; index < 8; ++index) crfBurst.u32Data_[index] = nop.toInt();
    crfBurst.u32Data_[0] =
        PIMCmd(PIMCmdType::FILL, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK).toInt();
    crfBurst.u32Data_[1] =
        PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A,
               PIMOpdType::ODD_BANK, 1, 0, 0, 0)
            .toInt();
    uint32_t registerTiles = wave.segmentCount / 8;
    crfBurst.u32Data_[2] =
        PIMCmd(PIMCmdType::JUMP, static_cast<int>(registerTiles - 1), 3).toInt();
    crfBurst.u32Data_[3] = PIMCmd(PIMCmdType::EXIT, 0).toInt();

    submit(true, pimAddress(channel, rank, 1, kPimRegRow, 0x4), "SPARSE_PROGRAM_CRF",
           &crfBurst, RequestClass::PIM_CONTROL, SparsePhase::PIM_COMPUTE, stats);
    barrier(channel);

    submit(true, pimAddress(channel, rank, 0, kPimRegRow, 0), "SPARSE_PIM_ON", &controlOn_,
           RequestClass::PIM_CONTROL, SparsePhase::PIM_COMPUTE, stats);
    barrier(channel);

    for (uint32_t tile = 0; tile < registerTiles; ++tile)
    {
        uint32_t row = wave.dataBaseRow + 2 * tile;
        for (uint32_t grf = 0; grf < 8; ++grf)
            submit(false, pimAddress(channel, rank, 0, row, grf), "SPARSE_FILL", &nullBurst_,
                   RequestClass::DENSE_VECTOR, SparsePhase::PIM_COMPUTE, stats, pimBlocks_);
        barrier(channel);
        for (uint32_t grf = 0; grf < 8; ++grf)
            submit(false, pimAddress(channel, rank, 1, row, grf), "SPARSE_MAC", &nullBurst_,
                   RequestClass::MATRIX_VALUE, SparsePhase::PIM_COMPUTE, stats, pimBlocks_);
        barrier(channel);
    }

    // Execute the exhausted JUMP and EXIT so the following write takes the HAB result path.
    submit(false, pimAddress(channel, rank, 0, wave.dataBaseRow, 0), "SPARSE_EXIT_CRF",
           &nullBurst_, RequestClass::PIM_CONTROL, SparsePhase::PIM_COMPUTE, stats, 0);
    barrier(channel);
    submit(true, pimAddress(channel, rank, 1, wave.resultRow, 0), "SPARSE_STORE_RESULT",
           &nullBurst_, RequestClass::RESULT, SparsePhase::PIM_COMPUTE, stats, pimBlocks_);
    barrier(channel);

    submit(true, pimAddress(channel, rank, 0, kPimRegRow, 0), "SPARSE_PIM_OFF", &controlOff_,
           RequestClass::PIM_CONTROL, SparsePhase::PIM_COMPUTE, stats);
    barrier(channel);
}

void HostGatherLockstepExecutor::queueCompute(SparseExecutionLayout& layout,
                                              SparseRunStats& stats)
{
    size_t nextWave = 0;
    for (uint32_t channel = 0; channel < channels_; ++channel)
    {
        for (uint32_t rank = 0; rank < ranks_; ++rank)
        {
            size_t groupStart = nextWave;
            while (nextWave < layout.waves.size() && layout.waves[nextWave].channel == channel &&
                   layout.waves[nextWave].rank == rank)
                ++nextWave;
            if (groupStart == nextWave)
                continue;

            parkIn(channel, rank, stats);
            enterHab(channel, rank, stats);
            for (size_t waveIndex = groupStart; waveIndex < nextWave; ++waveIndex)
                programWave(channel, rank, layout.waves[waveIndex], layout.crfBursts[waveIndex],
                            stats);
            exitHab(channel, rank, stats);
            parkOut(channel, rank, stats);
        }
    }
}

void HostGatherLockstepExecutor::queueReadback(const SparseExecutionLayout& layout,
                                               SparseExecutionLayout& mutableLayout,
                                               SparseRunStats& stats)
{
    for (const SparseLockstepWave& wave : layout.waves)
    {
        for (uint32_t block = 0; block < pimBlocks_; ++block)
        {
            if (wave.rowByBlock[block] < 0)
                continue;
            uint32_t rowId = static_cast<uint32_t>(wave.rowByBlock[block]);
            submit(false, address(wave.channel, wave.rank, block * 2 + 1, wave.resultRow, 0),
                   "SPARSE_READ_RESULT", &mutableLayout.resultBursts[rowId], RequestClass::RESULT,
                   SparsePhase::RESULT_READBACK, stats);
        }
    }
}

SparseExecutionResult HostGatherLockstepExecutor::run(const SparseMatrixCSR& matrix,
                                                      const SparseMappingPlan& plan,
                                                      const std::vector<fp16>& input)
{
#ifdef NO_STORAGE
    throw std::runtime_error("functional sparse execution requires a build without NO_STORAGE");
#endif
    SparseExecutionResult result;
    result.values.assign(matrix.numRows, convertF2H(0.0f));
    result.stats.memoryBefore = memory_->getStatsSnapshot();
    SparseExecutionLayout layout = buildLayout(matrix, plan);
    result.stats.workerScheduledSegments.assign(plan.workers.size(), 0);
    result.stats.workerEffectiveBankBursts.assign(plan.workers.size(), 0);

    for (uint32_t row = 0; row < matrix.numRows; ++row)
    {
        uint32_t segments = matrix.rowSegments(row);
        result.stats.usefulSegments += segments;
        result.stats.reloadRounds += segments > 0 ? (segments - 1) / 8 : 0;
        result.stats.grfOverwriteBursts += segments > 8 ? segments - 8 : 0;
    }
    for (const SparseLockstepWave& wave : layout.waves)
    {
        result.stats.scheduledSegments += static_cast<uint64_t>(pimBlocks_) * wave.segmentCount;
        size_t baseWorker = (static_cast<size_t>(wave.channel) * ranks_ + wave.rank) * pimBlocks_;
        for (uint32_t block = 0; block < pimBlocks_; ++block)
        {
            size_t worker = baseWorker + block;
            result.stats.workerScheduledSegments[worker] += wave.segmentCount;
            // Setup weight write, vector staging write, FILL, and MAC for every scheduled slot;
            // one result-store bank burst per wave, plus readback for active rows.
            result.stats.workerEffectiveBankBursts[worker] +=
                4 * static_cast<uint64_t>(wave.segmentCount) + 1;
            if (wave.rowByBlock[block] >= 0)
                result.stats.workerEffectiveBankBursts[worker]++;
        }
    }
    result.stats.wavePaddingSegments =
        result.stats.scheduledSegments - result.stats.usefulSegments;
    result.stats.usefulVectorBursts = result.stats.usefulSegments;
    result.stats.paddedVectorBursts = result.stats.wavePaddingSegments;
    result.stats.cpuIndexBytes = static_cast<uint64_t>(matrix.nnz()) * sizeof(uint32_t);
    result.stats.cpuGatherValueBytes = static_cast<uint64_t>(matrix.nnz()) * sizeof(fp16);

    queueMatrixSetup(matrix, plan, layout, result.stats);
    result.stats.cycles.matrixSetup = drain();

    queueVectorStaging(plan, input, layout, result.stats);
    result.stats.cycles.vectorStaging = drain();

    queueCompute(layout, result.stats);
    result.stats.cycles.pimCompute = drain();

    queueReadback(layout, layout, result.stats);
    result.stats.cycles.resultReadback = drain();

    result.stats.cycles.recurringSpmv = result.stats.cycles.vectorStaging +
                                        result.stats.cycles.pimCompute +
                                        result.stats.cycles.resultReadback;
    result.stats.cycles.endToEnd = result.stats.cycles.matrixSetup +
                                   result.stats.cycles.recurringSpmv;
    for (uint32_t row = 0; row < matrix.numRows; ++row)
        result.values[row] = layout.resultBursts[row].fp16ReduceSum();
    result.stats.memoryAfter = memory_->getStatsSnapshot();
    return result;
}

}  // namespace DRAMSim

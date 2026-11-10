/***************************************************************************************************
 * Reproducible host-side sparse matrix/vector multiplication benchmarks.
 *
 * These tests use the existing HBM path only.  They establish a pre-PIM memory-system baseline;
 * PIMSimulator does not model host CPU instructions or caches, so cycles reported here are memory
 * service cycles rather than complete CPU execution time.
 **************************************************************************************************/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "MultiChannelMemorySystem.h"
#include "SparsePIMEnergy.h"
#include "gtest/gtest.h"

using namespace DRAMSim;

namespace
{

struct CsrMatrix
{
    uint32_t rows;
    uint32_t cols;
    std::vector<uint32_t> row_ptr;
    std::vector<uint32_t> col_idx;
    std::vector<float> values;
};

struct BenchmarkSpec
{
    std::string group;
    std::string name;
    uint32_t rows;
    uint32_t cols;
    std::string pattern;
    uint32_t nnz_per_row;
    std::string format;
    std::string system_config;
    uint32_t block_size;
    uint32_t seed;
};

struct TrafficPlan
{
    std::vector<uint64_t> packed_stream_bytes;
    std::vector<uint32_t> input_columns;
    uint64_t stored_entries = 0;
};

struct BenchmarkResult
{
    uint64_t cycles = 0;
    uint64_t reads = 0;
    uint64_t writes = 0;
    uint64_t active_channels = 0;
    uint64_t max_channel_transactions = 0;
    double average_channel_transactions = 0.0;
    double channel_imbalance = 0.0;
    uint64_t energy = 0;
    uint64_t stored_entries = 0;
};

uint64_t hash64(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

uint32_t envUnsigned(const char* name, uint32_t default_value)
{
    const char* value = std::getenv(name);
    return value == nullptr ? default_value : static_cast<uint32_t>(std::stoul(value));
}

std::string envString(const char* name, const std::string& default_value)
{
    const char* value = std::getenv(name);
    return value == nullptr ? default_value : value;
}

BenchmarkSpec specFromEnvironment()
{
    return {envString("SPMV_GROUP", "smoke"),
            envString("SPMV_NAME", "default"),
            envUnsigned("SPMV_ROWS", 256),
            envUnsigned("SPMV_COLS", 256),
            envString("SPMV_PATTERN", "uniform"),
            envUnsigned("SPMV_NNZ_PER_ROW", 8),
            envString("SPMV_FORMAT", "csr"),
            envString("SPMV_SYSTEM_CONFIG", "system_hbm.ini"),
            envUnsigned("SPMV_BLOCK_SIZE", 4),
            envUnsigned("SPMV_SEED", 570)};
}

uint32_t entriesInRow(const BenchmarkSpec& spec, uint32_t row)
{
    if (spec.pattern == "moderate_skew")
    {
        const uint32_t limit = std::max(2U, 2 * spec.nnz_per_row);
        return 1 + hash64(spec.seed + row) % limit;
    }
    if (spec.pattern == "heavy_skew" || spec.pattern == "social_proxy")
    {
        return row % 32 == 0 ? std::min(spec.cols, 16 * spec.nnz_per_row)
                             : std::max(1U, spec.nnz_per_row / 2);
    }
    if (spec.pattern == "road_proxy")
    {
        return 2 + hash64(spec.seed + row) % std::max(1U, spec.nnz_per_row - 1);
    }
    return std::min(spec.cols, spec.nnz_per_row);
}

uint32_t candidateColumn(const BenchmarkSpec& spec, uint32_t row, uint32_t entry,
                         uint32_t attempt, uint32_t row_entries)
{
    if (spec.pattern == "banded" || spec.pattern == "sequential" ||
        spec.pattern == "road_proxy")
    {
        const int64_t offset = static_cast<int64_t>(entry) - row_entries / 2;
        return static_cast<uint32_t>((static_cast<int64_t>(row) + offset + spec.cols) % spec.cols);
    }
    if (spec.pattern == "block" || spec.pattern == "circuit_proxy")
    {
        constexpr uint32_t locality_block = 32;
        const uint32_t block_start = (row / locality_block) * locality_block;
        if (spec.pattern == "circuit_proxy" && entry % 5 == 0)
        {
            return hash64(spec.seed + row * 131ULL + entry + attempt) % spec.cols;
        }
        return (block_start + entry + attempt) % spec.cols;
    }
    if (spec.pattern == "hotset" || spec.pattern == "social_proxy")
    {
        const uint32_t hot_columns = std::min(spec.cols, std::max(16U, spec.cols / 16));
        return hash64(spec.seed + row * 131ULL + entry * 17ULL + attempt) % hot_columns;
    }
    return hash64(spec.seed + row * 0x10001ULL + entry * 131ULL + attempt) % spec.cols;
}

CsrMatrix makeMatrix(const BenchmarkSpec& spec)
{
    CsrMatrix matrix{spec.rows, spec.cols, {0}, {}, {}};
    for (uint32_t row = 0; row < spec.rows; ++row)
    {
        const uint32_t row_entries = std::min(spec.cols, entriesInRow(spec, row));
        std::vector<uint32_t> columns;
        columns.reserve(row_entries);
        for (uint32_t entry = 0; entry < row_entries; ++entry)
        {
            uint32_t attempt = 0;
            uint32_t column = 0;
            do
            {
                column = candidateColumn(spec, row, entry, attempt++, row_entries);
            } while (std::find(columns.begin(), columns.end(), column) != columns.end());
            columns.push_back(column);
        }

        for (uint32_t entry = 0; entry < columns.size(); ++entry)
        {
            matrix.col_idx.push_back(columns[entry]);
            matrix.values.push_back(static_cast<float>((row + entry + spec.seed) % 17 + 1) / 17.0F);
        }
        matrix.row_ptr.push_back(static_cast<uint32_t>(matrix.col_idx.size()));
    }
    return matrix;
}

std::vector<float> spmvCsr(const CsrMatrix& matrix, const std::vector<float>& input)
{
    std::vector<float> output(matrix.rows, 0.0F);
    for (uint32_t row = 0; row < matrix.rows; ++row)
    {
        for (uint32_t index = matrix.row_ptr[row]; index < matrix.row_ptr[row + 1]; ++index)
        {
            output[row] += matrix.values[index] * input[matrix.col_idx[index]];
        }
    }
    return output;
}

std::vector<float> spmvDenseReference(const CsrMatrix& matrix, const std::vector<float>& input)
{
    std::vector<float> dense(static_cast<size_t>(matrix.rows) * matrix.cols, 0.0F);
    for (uint32_t row = 0; row < matrix.rows; ++row)
    {
        for (uint32_t index = matrix.row_ptr[row]; index < matrix.row_ptr[row + 1]; ++index)
        {
            dense[static_cast<size_t>(row) * matrix.cols + matrix.col_idx[index]] =
                matrix.values[index];
        }
    }

    std::vector<float> output(matrix.rows, 0.0F);
    for (uint32_t row = 0; row < matrix.rows; ++row)
    {
        for (uint32_t col = 0; col < matrix.cols; ++col)
        {
            output[row] += dense[static_cast<size_t>(row) * matrix.cols + col] * input[col];
        }
    }
    return output;
}

TrafficPlan makeTrafficPlan(const CsrMatrix& matrix, const BenchmarkSpec& spec)
{
    TrafficPlan plan;
    const uint64_t nnz = matrix.values.size();

    if (spec.format == "csr")
    {
        plan.packed_stream_bytes = {(matrix.rows + 1ULL) * sizeof(uint32_t),
                                    nnz * sizeof(uint32_t), nnz * sizeof(float)};
        plan.input_columns = matrix.col_idx;
        plan.stored_entries = nnz;
    }
    else if (spec.format == "coo")
    {
        plan.packed_stream_bytes = {nnz * sizeof(uint32_t), nnz * sizeof(uint32_t),
                                    nnz * sizeof(float)};
        plan.input_columns = matrix.col_idx;
        plan.stored_entries = nnz;
    }
    else if (spec.format == "ell")
    {
        uint64_t max_row_entries = 0;
        for (uint32_t row = 0; row < matrix.rows; ++row)
        {
            max_row_entries = std::max<uint64_t>(
                max_row_entries, matrix.row_ptr[row + 1] - matrix.row_ptr[row]);
        }
        plan.stored_entries = matrix.rows * max_row_entries;
        plan.packed_stream_bytes = {plan.stored_entries * sizeof(uint32_t),
                                    plan.stored_entries * sizeof(float)};
        for (uint32_t row = 0; row < matrix.rows; ++row)
        {
            const uint32_t begin = matrix.row_ptr[row];
            const uint32_t end = matrix.row_ptr[row + 1];
            for (uint64_t slot = 0; slot < max_row_entries; ++slot)
            {
                plan.input_columns.push_back(slot < end - begin ? matrix.col_idx[begin + slot] : 0);
            }
        }
    }
    else if (spec.format == "bcsr")
    {
        const uint32_t block = std::max(1U, spec.block_size);
        const uint32_t block_rows = (matrix.rows + block - 1) / block;
        uint64_t blocks = 0;
        for (uint32_t block_row = 0; block_row < block_rows; ++block_row)
        {
            std::set<uint32_t> block_columns;
            const uint32_t row_end = std::min(matrix.rows, (block_row + 1) * block);
            for (uint32_t row = block_row * block; row < row_end; ++row)
            {
                for (uint32_t index = matrix.row_ptr[row]; index < matrix.row_ptr[row + 1]; ++index)
                {
                    block_columns.insert(matrix.col_idx[index] / block);
                }
            }
            blocks += block_columns.size();
            for (uint32_t block_col : block_columns)
            {
                for (uint32_t row_in_block = 0; row_in_block < block; ++row_in_block)
                {
                    for (uint32_t col_in_block = 0; col_in_block < block; ++col_in_block)
                    {
                        plan.input_columns.push_back(
                            std::min(matrix.cols - 1, block_col * block + col_in_block));
                    }
                }
            }
        }
        plan.stored_entries = blocks * block * block;
        plan.packed_stream_bytes = {(block_rows + 1ULL) * sizeof(uint32_t),
                                    blocks * sizeof(uint32_t),
                                    plan.stored_entries * sizeof(float)};
    }
    else
    {
        throw std::invalid_argument("Unsupported sparse format: " + spec.format);
    }
    return plan;
}

uint64_t alignUp(uint64_t address, uint64_t alignment)
{
    return ((address + alignment - 1) / alignment) * alignment;
}

void addPackedTraffic(const std::shared_ptr<MultiChannelMemorySystem>& memory, bool is_write,
                      uint64_t base_address, uint64_t size_in_bytes, uint64_t burst_bytes,
                      BurstType* data)
{
    for (uint64_t offset = 0; offset < size_in_bytes; offset += burst_bytes)
    {
        ASSERT_TRUE(memory->addTransaction(is_write, base_address + offset, data));
    }
}

BenchmarkResult runMemoryBenchmark(const CsrMatrix& matrix, const BenchmarkSpec& spec)
{
    const TrafficPlan plan = makeTrafficPlan(matrix, spec);
    auto memory = std::make_shared<MultiChannelMemorySystem>(
        "ini/HBM2_samsung_2M_16B_x64.ini", spec.system_config, ".", spec.name, 256 * 64);
    const uint64_t burst_bytes =
        getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") * getConfigParam(UINT, "BL") / 8;

    BurstType data;
    uint64_t next_base = 0;
    for (uint64_t stream_bytes : plan.packed_stream_bytes)
    {
        next_base = alignUp(next_base, burst_bytes);
        addPackedTraffic(memory, false, next_base, stream_bytes, burst_bytes, &data);
        next_base += stream_bytes;
    }

    const uint64_t input_base = alignUp(next_base, burst_bytes);
    for (uint32_t col : plan.input_columns)
    {
        const uint64_t input_offset = static_cast<uint64_t>(col) * sizeof(float);
        const uint64_t input_burst = (input_offset / burst_bytes) * burst_bytes;
        EXPECT_TRUE(memory->addTransaction(false, input_base + input_burst, &data));
    }

    const uint64_t output_base = alignUp(input_base + matrix.cols * sizeof(float), burst_bytes);
    addPackedTraffic(memory, true, output_base, matrix.rows * sizeof(float), burst_bytes, &data);

    BenchmarkResult result;
    result.stored_entries = plan.stored_entries;
    while (memory->hasPendingTransactions())
    {
        memory->update();
        ++result.cycles;
    }

    for (auto* channel : memory->channels)
    {
        const uint64_t channel_reads = channel->memoryController->totalReads;
        const uint64_t channel_writes = channel->memoryController->totalWrites;
        const uint64_t channel_transactions = channel_reads + channel_writes;
        result.reads += channel_reads;
        result.writes += channel_writes;
        result.active_channels += channel_transactions > 0;
        result.max_channel_transactions =
            std::max(result.max_channel_transactions, channel_transactions);
        for (uint64_t value : channel->memoryController->burstEnergy)
        {
            result.energy += value;
        }
        for (uint64_t value : channel->memoryController->actpreEnergy)
        {
            result.energy += value;
        }
    }
    result.average_channel_transactions =
        static_cast<double>(result.reads + result.writes) / memory->channels.size();
    result.channel_imbalance = result.average_channel_transactions == 0.0
                                   ? 0.0
                                   : result.max_channel_transactions /
                                         result.average_channel_transactions;
    return result;
}

void printResult(const CsrMatrix& matrix, const BenchmarkSpec& spec,
                 const BenchmarkResult& result)
{
    const double density = static_cast<double>(matrix.values.size()) /
                           (static_cast<double>(matrix.rows) * matrix.cols);
    std::cout << "SPMV_RESULT,"
              << spec.group << ',' << spec.name << ',' << spec.pattern << ',' << spec.format << ','
              << spec.system_config << ',' << matrix.rows << ',' << matrix.cols << ','
              << matrix.values.size() << ',' << density << ',' << result.stored_entries << ','
              << result.cycles << ',' << result.reads << ',' << result.writes << ','
              << result.active_channels << ',' << result.max_channel_transactions << ','
              << result.average_channel_transactions << ',' << result.channel_imbalance << ','
              << result.energy << std::endl;
}

TEST(SparseMatrixBaseline, csr_spmv_correctness)
{
    const BenchmarkSpec spec{"smoke", "csr_correctness", 256, 256, "uniform", 8,
                             "csr", "system_hbm.ini", 4, 570};
    const CsrMatrix matrix = makeMatrix(spec);
    std::vector<float> input(matrix.cols);
    for (uint32_t col = 0; col < matrix.cols; ++col)
    {
        input[col] = static_cast<float>(static_cast<int>(col % 13) - 6) / 7.0F;
    }

    const std::vector<float> expected = spmvDenseReference(matrix, input);
    const std::vector<float> actual = spmvCsr(matrix, input);
    ASSERT_EQ(actual.size(), expected.size());
    for (uint32_t row = 0; row < matrix.rows; ++row)
    {
        EXPECT_NEAR(actual[row], expected[row], 1.0e-5F);
    }

    const BenchmarkResult result = runMemoryBenchmark(matrix, spec);
    printResult(matrix, spec, result);
    EXPECT_GT(result.cycles, 0U);
    EXPECT_GT(result.reads, 0U);
    EXPECT_GT(result.writes, 0U);
}

TEST(SparseMatrixBenchmark, configured_case)
{
    const BenchmarkSpec spec = specFromEnvironment();
    ASSERT_GT(spec.rows, 0U);
    ASSERT_GT(spec.cols, 0U);
    ASSERT_GT(spec.nnz_per_row, 0U);

    const CsrMatrix matrix = makeMatrix(spec);
    ASSERT_EQ(matrix.row_ptr.size(), matrix.rows + 1ULL);
    ASSERT_EQ(matrix.col_idx.size(), matrix.values.size());
    ASSERT_LE(matrix.values.size(), static_cast<uint64_t>(matrix.rows) * matrix.cols);

    const BenchmarkResult result = runMemoryBenchmark(matrix, spec);
    printResult(matrix, spec, result);
    EXPECT_GT(result.cycles, 0U);
    EXPECT_GT(result.reads, 0U);
    EXPECT_GT(result.writes, 0U);
}

}  // namespace

// Opt-in PIM execution. The host-traffic baseline above stays unchanged.
namespace
{
struct PimResult
{
    SparsePIM::Stats stats;
    MemoryController::Activity activity;
    std::vector<float> output;
    double dramPJ = 0, pimPJ = 0, runtimeNS = 0;
    std::string energyModel;
    unsigned macLatency = 1, physicalBlocks = 0;
};

MemoryController::Activity activitySnapshot(const MultiChannelMemorySystem& memory)
{
    MemoryController::Activity total;
    for (const auto* channel : memory.channels)
    {
        const auto& a = channel->memoryController->activity;
        total.reads += a.reads; total.writes += a.writes;
        total.activates += a.activates; total.precharges += a.precharges;
        total.refreshes += a.refreshes;
    }
    return total;
}

void drainSparseMemory(MultiChannelMemorySystem& memory)
{
    const uint64_t limit = envUnsigned("SPMV_MAX_CYCLES", 100000000);
    uint64_t cycles = 0;
    while (memory.hasPendingTransactions())
    {
        if (++cycles > limit) throw std::runtime_error("Sparse PIM watchdog exceeded SPMV_MAX_CYCLES");
        memory.update();
    }
}

PimResult runSparsePIM(const CsrMatrix& matrix, const BenchmarkSpec& spec,
                       SparsePIMPolicy policy, unsigned macLatency = 1)
{
    // Keep one rank per channel when varying channel count. The host baseline's
    // fixed 16 GiB capacity is retained only in its original path.
    auto& db = ConfigurationDB::getDB();
    db.initialize();
    db.updatefromFile("ini/HBM2_samsung_2M_16B_x64.ini");
    db.updatefromFile(spec.system_config);
    const uint64_t capacityMB = (uint64_t(getConfigParam(UINT, "NUM_ROWS")) *
        getConfigParam(UINT, "NUM_COLS") * getConfigParam(UINT, "NUM_BANKS") *
        getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") / 8 >> 20) * getConfigParam(UINT, "NUM_CHANS");
    auto memory = std::make_shared<MultiChannelMemorySystem>(
        "ini/HBM2_samsung_2M_16B_x64.ini", spec.system_config, ".", spec.name + "_pim", capacityMB);
    const unsigned channels = memory->channels.size();
    const unsigned ranks = getConfigParam(UINT, "NUM_RANKS");
    const unsigned blocksPerRank = getConfigParam(UINT, "NUM_PIM_BLOCKS");
    const unsigned count = channels * ranks * blocksPerRank;
    SparsePIMJob job;
    job.columns = matrix.cols;
    job.address = SparsePIM::bankAddress;
    job.partitions.resize(count);
    std::vector<std::vector<uint32_t>> rowLists(count);
    // Static placement only: no per-operation schedules, skips, or issue times.
    for (uint32_t row = 0; row < matrix.rows; ++row) rowLists[row % count].push_back(row);
    std::vector<std::vector<BurstType>> images(count);
    std::vector<float> input(matrix.cols);
    for (uint32_t col = 0; col < matrix.cols; ++col)
        input[col] = static_cast<float>(static_cast<int>(col % 13) - 6) / 7.0F;
    const auto energy = SparsePIMEnergyModel::load(envString(
        "SPMV_ENERGY_MODEL", "data/sparse_matrices/pim_energy_example.txt"));

    for (unsigned id = 0; id < count; ++id)
    {
        auto& p = job.partitions[id];
        p.channel = id / (ranks * blocksPerRank);
        p.rank = (id / blocksPerRank) % ranks;
        p.block = id % blocksPerRank;
        p.rows = rowLists[id].size();
        for (auto row : rowLists[id]) p.entries += matrix.row_ptr[row + 1] - matrix.row_ptr[row];
        p.rowPtr = 0;
        p.columns = alignUp((p.rows + 1ULL) * 4, sizeof(BurstType));
        p.values = alignUp(p.columns + p.entries * 4ULL, sizeof(BurstType));
        p.input = alignUp(p.values + p.entries * 4ULL, sizeof(BurstType));
        p.output = alignUp(p.input + matrix.cols * 4ULL, sizeof(BurstType));
        if (!p.rows) continue;
        // Stable burst buffers live until every setup transaction has completed.
        images[id].resize((p.output + p.rows * sizeof(BurstType)) / sizeof(BurstType));
        auto putU32 = [&](uint64_t offset, uint32_t value) {
            images[id][offset / sizeof(BurstType)].u32Data_[(offset % sizeof(BurstType)) / 4] = value;
        };
        auto putFloat = [&](uint64_t offset, float value) {
            images[id][offset / sizeof(BurstType)].fp32Data_[(offset % sizeof(BurstType)) / 4] = value;
        };
        uint32_t cursor = 0, localRow = 0;
        for (auto row : rowLists[id])
        {
            putU32(p.rowPtr + 4ULL * localRow++, cursor);
            for (uint32_t entry = matrix.row_ptr[row]; entry < matrix.row_ptr[row + 1]; ++entry)
            {
                putU32(p.columns + 4ULL * cursor, matrix.col_idx[entry]);
                putFloat(p.values + 4ULL * cursor++, matrix.values[entry]);
            }
        }
        putU32(p.rowPtr + 4ULL * localRow, cursor);
        for (uint32_t col = 0; col < matrix.cols; ++col) putFloat(p.input + 4ULL * col, input[col]);
        for (size_t slot = 0; slot < images[id].size(); ++slot)
        {
            const uint64_t addr = job.address(p, slot * sizeof(BurstType));
            memory->addTransaction(true, addr, &images[id][slot]);
        }
    }
    drainSparseMemory(*memory);
    const auto before = activitySnapshot(*memory);
    memory->submitSparsePIM(job, policy, macLatency);
    // This is the entire timed execution loop. The test never schedules blocks.
    drainSparseMemory(*memory);
    PimResult result;
    result.stats = memory->sparsePIM()->stats();
    const auto after = activitySnapshot(*memory);
    result.activity.reads = after.reads - before.reads;
    result.activity.writes = after.writes - before.writes;
    result.activity.activates = after.activates - before.activates;
    result.activity.precharges = after.precharges - before.precharges;
    result.activity.refreshes = after.refreshes - before.refreshes;
    result.runtimeNS = result.stats.cycles * getConfigParam(FLOAT, "tCK");
    result.dramPJ = energy.dramPJ(result.activity, result.runtimeNS, channels * ranks);
    result.pimPJ = energy.pimPJ(result.stats, result.runtimeNS, count,
                               policy == SparsePIMPolicy::Independent);
    result.energyModel = energy.name;
    result.macLatency = macLatency; result.physicalBlocks = count;

    // Read actual simulated output after the timed region; never use host SpMV as output.
    std::vector<BurstType> readback(matrix.rows);
    for (unsigned id = 0; id < count; ++id)
        for (size_t row = 0; row < rowLists[id].size(); ++row)
        {
            const auto& p = job.partitions[id];
            memory->addTransaction(false, job.address(p, p.output + row * sizeof(BurstType)),
                                   &readback[rowLists[id][row]]);
        }
    drainSparseMemory(*memory);
    result.output.resize(matrix.rows);
    const auto expected = spmvCsr(matrix, input);
    for (uint32_t row = 0; row < matrix.rows; ++row)
    {
        result.output[row] = readback[row].fp32Data_[0];
        EXPECT_NEAR(result.output[row], expected[row], 1.0e-5F) << "row=" << row;
    }
    EXPECT_EQ(result.stats.macs, matrix.values.size());
    EXPECT_EQ(result.stats.completedRows, matrix.rows);
    return result;
}

void printPimResult(const CsrMatrix& matrix, const BenchmarkSpec& spec,
                    const PimResult& r, const std::string& mode)
{
    const double total = r.dramPJ + r.pimPJ;
    std::cout << std::setprecision(12) << "SPMV_PIM_RESULT," << spec.group << ',' << spec.name << ','
              << spec.pattern << ',' << spec.format << ",csr," << spec.system_config << ','
              << matrix.rows << ',' << matrix.cols << ',' << matrix.values.size() << ',' << mode << ','
              << spec.seed << ',' << r.macLatency << ',' << r.physicalBlocks << ',' << r.stats.cycles << ',' << r.runtimeNS << ',' << r.activity.reads << ',' << r.activity.writes << ','
              << r.stats.macs << ',' << r.stats.memoryWaitCycles << ',' << r.stats.barrierCycles << ','
              << r.stats.maxRowLead << ',' << r.dramPJ << ',' << r.pimPJ << ',' << total << ','
              << (r.runtimeNS ? total / r.runtimeNS : 0) << ',' << r.energyModel << std::endl;
}

TEST(SparsePIMBenchmark, configured_case)
{
    const auto spec = specFromEnvironment();
    ASSERT_GT(spec.rows, 0U); ASSERT_GT(spec.cols, 0U); ASSERT_GT(spec.nnz_per_row, 0U);
    const auto matrix = makeMatrix(spec);
    const auto mode = envString("SPMV_PIM_CONTROL", "lockstep");
    ASSERT_TRUE(mode == "lockstep" || mode == "independent");
    const auto result = runSparsePIM(matrix, spec,
        mode == "independent" ? SparsePIMPolicy::Independent : SparsePIMPolicy::Lockstep,
        envUnsigned("SPMV_PIM_MAC_CYCLES", 1));
    printPimResult(matrix, spec, result, mode);
}

TEST(SparsePIMControl, uneven_rows_execute_ahead_and_match_lockstep)
{
    BenchmarkSpec spec{"test", "uneven", 32, 64, "uniform", 1,
                       "csr", "system_hbm_1ch.ini", 4, 570};
    // Two waves: long rows rotate across blocks, exposing row synchronization costs.
    CsrMatrix matrix{spec.rows, spec.cols, {0}, {}, {}};
    for (uint32_t row = 0; row < matrix.rows; ++row)
    {
        const unsigned length = row % 8 == (row / 8) % 8 ? 64 : 1;
        for (unsigned col = 0; col < length; ++col)
        {
            matrix.col_idx.push_back(col);
            matrix.values.push_back(0.5F);
        }
        matrix.row_ptr.push_back(matrix.values.size());
    }
    const auto baseline = runSparsePIM(matrix, spec, SparsePIMPolicy::Lockstep, 3);
    const auto independent = runSparsePIM(matrix, spec, SparsePIMPolicy::Independent, 3);
    EXPECT_EQ(baseline.output, independent.output);
    EXPECT_EQ(baseline.activity.reads, independent.activity.reads);
    EXPECT_EQ(baseline.activity.writes, independent.activity.writes);
    EXPECT_GT(independent.stats.maxRowLead, 1U);
    EXPECT_LT(independent.stats.cycles, baseline.stats.cycles);
    EXPECT_EQ(independent.stats.barrierCycles, 0U);
    printPimResult(matrix, spec, baseline, "lockstep");
    printPimResult(matrix, spec, independent, "independent");
}

TEST(SparsePIMControl, empty_rows_tail_and_signed_values)
{
    BenchmarkSpec spec{"test", "empty_tail", 19, 11, "uniform", 1,
                       "csr", "system_hbm_1ch.ini", 4, 570};
    CsrMatrix matrix{19, 11, {0}, {}, {}};
    for (unsigned row = 0; row < matrix.rows; ++row)
    {
        if (row % 3 == 0)
        {
            matrix.col_idx.push_back(row % matrix.cols);
            matrix.values.push_back(row % 2 ? -2.0F : 3.0F);
        }
        matrix.row_ptr.push_back(matrix.values.size());
    }
    const auto a = runSparsePIM(matrix, spec, SparsePIMPolicy::Lockstep);
    const auto b = runSparsePIM(matrix, spec, SparsePIMPolicy::Independent);
    EXPECT_EQ(a.output, b.output);
    matrix.col_idx.clear(); matrix.values.clear();
    std::fill(matrix.row_ptr.begin(), matrix.row_ptr.end(), 0);
    const auto empty = runSparsePIM(matrix, spec, SparsePIMPolicy::Independent);
    EXPECT_EQ(empty.output, std::vector<float>(matrix.rows, 0));
}
} // namespace

namespace
{
TEST(SparsePIMControl, device_completion_and_shared_read_addresses)
{
    MultiChannelMemorySystem memory("ini/HBM2_samsung_2M_16B_x64.ini",
        "system_hbm_1ch.ini", ".", "completion_test", 256);
    BurstType stored, first, second;
    stored.u32Data_[0] = 0x12345678;
    unsigned completions = 0;
    auto* write = new Transaction(DATA_WRITE, 0, &stored);
    write->onComplete = [&]() { ++completions; };
    ASSERT_TRUE(memory.addTransaction(write));
    drainSparseMemory(memory);
    ASSERT_EQ(completions, 1U);
    for (auto* buffer : {&first, &second})
    {
        auto* read = new Transaction(DATA_READ, 0, buffer);
        read->onComplete = [&, buffer]() {
            EXPECT_EQ(buffer->u32Data_[0], 0x12345678U);
            ++completions;
        };
        ASSERT_TRUE(memory.addTransaction(read));
    }
    drainSparseMemory(memory);
    EXPECT_EQ(completions, 3U);
    SparsePIMJob job;
    job.columns = 1;
    job.partitions.resize(2); // Both claim the same hardware block.
    EXPECT_THROW(memory.submitSparsePIM(job, SparsePIMPolicy::Independent), std::invalid_argument);
    job.partitions.resize(1);
    job.partitions[0].rows = 1;
    EXPECT_THROW(memory.submitSparsePIM(job, SparsePIMPolicy::Independent, 0), std::invalid_argument);
    // No device progress is needed to detect a second submission while busy.
    memory.submitSparsePIM(job, SparsePIMPolicy::Independent);
    EXPECT_TRUE(memory.hasPendingTransactions());
    EXPECT_THROW(memory.submitSparsePIM(job, SparsePIMPolicy::Independent), std::logic_error);
}

TEST(SparsePIMControl, energy_units_and_independent_control_cost)
{
    SparsePIMEnergyModel model;
    model.readPJ = 2; model.macPJ = 3; model.instructionPJ = 0.5;
    model.rankMW = 4; model.blockMW = 5; model.independentMW = 1;
    MemoryController::Activity activity;
    activity.reads = 7;
    SparsePIM::Stats stats;
    stats.macs = 2; stats.instructions = 4;
    EXPECT_DOUBLE_EQ(model.dramPJ(activity, 10, 2), 94); // 7*2 + 10ns*2*4mW.
    EXPECT_DOUBLE_EQ(model.pimPJ(stats, 10, 2, false), 108);
    EXPECT_DOUBLE_EQ(model.pimPJ(stats, 10, 2, true), 128);
    EXPECT_THROW(SparsePIMEnergyModel::load("/nonexistent/pim_energy_model"), std::invalid_argument);
}
} // namespace

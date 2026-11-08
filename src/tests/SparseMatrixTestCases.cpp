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
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "MultiChannelMemorySystem.h"
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
    bool sparse_scheduler = true;
};

struct PackedStream
{
    uint64_t bytes;
    SparseRequestType request_type;
};

struct TrafficPlan
{
    std::vector<PackedStream> packed_streams;
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
    uint64_t sparse_scheduled = 0;
    uint64_t reordered = 0;
    uint64_t row_hit_admissions = 0;
    uint64_t starvation_promotions = 0;
    uint64_t queue_wait = 0;
    uint64_t activates = 0;
    uint64_t precharges = 0;
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
            envUnsigned("SPMV_SEED", 570),
            envString("SPMV_SCHEDULER", "sparse") != "baseline"};
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
        plan.packed_streams = {{(matrix.rows + 1ULL) * sizeof(uint32_t), SPARSE_METADATA},
                               {nnz * sizeof(uint32_t), SPARSE_INDEX},
                               {nnz * sizeof(float), SPARSE_VALUE}};
        plan.input_columns = matrix.col_idx;
        plan.stored_entries = nnz;
    }
    else if (spec.format == "coo")
    {
        plan.packed_streams = {{nnz * sizeof(uint32_t), SPARSE_METADATA},
                               {nnz * sizeof(uint32_t), SPARSE_INDEX},
                               {nnz * sizeof(float), SPARSE_VALUE}};
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
        plan.packed_streams = {{plan.stored_entries * sizeof(uint32_t), SPARSE_INDEX},
                               {plan.stored_entries * sizeof(float), SPARSE_VALUE}};
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
        plan.packed_streams = {{(block_rows + 1ULL) * sizeof(uint32_t), SPARSE_METADATA},
                               {blocks * sizeof(uint32_t), SPARSE_INDEX},
                               {plan.stored_entries * sizeof(float), SPARSE_VALUE}};
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
                      BurstType* data, SparseRequestType request_type,
                      uint64_t dependency_base = 0)
{
    for (uint64_t offset = 0; offset < size_in_bytes; offset += burst_bytes)
    {
        auto* transaction = new Transaction(is_write ? DATA_WRITE : DATA_READ,
                                            base_address + offset, data);
        transaction->setSparseMetadata(request_type, dependency_base + offset / burst_bytes);
        ASSERT_TRUE(memory->addTransaction(transaction));
    }
}

BenchmarkResult runMemoryBenchmark(const CsrMatrix& matrix, const BenchmarkSpec& spec)
{
    const TrafficPlan plan = makeTrafficPlan(matrix, spec);
    auto memory = std::make_shared<MultiChannelMemorySystem>(
        "ini/HBM2_samsung_2M_16B_x64.ini", spec.system_config, ".", spec.name, 256 * 64);
    memory->setSparseSchedulingEnabled(spec.sparse_scheduler);
    const uint64_t burst_bytes =
        getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") * getConfigParam(UINT, "BL") / 8;

    BurstType data;
    uint64_t next_base = 0;
    for (const PackedStream& stream : plan.packed_streams)
    {
        next_base = alignUp(next_base, burst_bytes);
        addPackedTraffic(memory, false, next_base, stream.bytes, burst_bytes, &data,
                         stream.request_type,
                         stream.request_type == SPARSE_METADATA ? (1ULL << 63) : 0);
        next_base += stream.bytes;
    }

    const uint64_t input_base = alignUp(next_base, burst_bytes);
    for (size_t entry = 0; entry < plan.input_columns.size(); ++entry)
    {
        const uint32_t col = plan.input_columns[entry];
        const uint64_t input_offset = static_cast<uint64_t>(col) * sizeof(float);
        const uint64_t input_burst = (input_offset / burst_bytes) * burst_bytes;
        auto* transaction = new Transaction(DATA_READ, input_base + input_burst, &data);
        transaction->setSparseMetadata(SPARSE_VECTOR,
                                       entry / std::max<uint64_t>(1, burst_bytes / sizeof(float)));
        EXPECT_TRUE(memory->addTransaction(transaction));
    }

    const uint64_t output_base = alignUp(input_base + matrix.cols * sizeof(float), burst_bytes);
    addPackedTraffic(memory, true, output_base, matrix.rows * sizeof(float), burst_bytes, &data,
                     SPARSE_OUTPUT);

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
        const auto& scheduler = channel->memoryController->getSparseSchedulerStats();
        result.sparse_scheduled += scheduler.sparseTransactions;
        result.reordered += scheduler.reorderedTransactions;
        result.row_hit_admissions += scheduler.rowHitAdmissions;
        result.starvation_promotions += scheduler.starvationPromotions;
        result.queue_wait += scheduler.totalQueueWait;
        result.activates += channel->memoryController->getTotalActivates();
        result.precharges += channel->memoryController->getTotalPrecharges();
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
              << result.energy << ',' << (spec.sparse_scheduler ? "sparse" : "baseline") << ','
              << result.sparse_scheduled << ',' << result.reordered << ','
              << result.row_hit_admissions << ',' << result.starvation_promotions << ','
              << (result.sparse_scheduled == 0
                      ? 0.0
                      : static_cast<double>(result.queue_wait) / result.sparse_scheduled)
              << ',' << result.activates << ',' << result.precharges << std::endl;
}

TEST(SparseMatrixBaseline, csr_spmv_correctness)
{
    const BenchmarkSpec spec{"smoke", "csr_correctness", 256, 256, "uniform", 8,
                             "csr", "system_hbm.ini", 4, 570, true};
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
    EXPECT_GT(result.sparse_scheduled, 0U);
    EXPECT_GT(result.reordered, 0U);
}

TEST(SparseAwareScheduler, preserves_transaction_counts)
{
    BenchmarkSpec baseline{"smoke", "scheduler_baseline", 128, 256, "hotset", 16,
                           "csr", "system_hbm.ini", 4, 570, false};
    BenchmarkSpec scheduled = baseline;
    scheduled.name = "scheduler_enabled";
    scheduled.sparse_scheduler = true;
    const CsrMatrix matrix = makeMatrix(baseline);

    const BenchmarkResult baseline_result = runMemoryBenchmark(matrix, baseline);
    const BenchmarkResult scheduled_result = runMemoryBenchmark(matrix, scheduled);
    EXPECT_EQ(scheduled_result.reads, baseline_result.reads);
    EXPECT_EQ(scheduled_result.writes, baseline_result.writes);
    EXPECT_GT(scheduled_result.sparse_scheduled, 0U);
    EXPECT_GT(scheduled_result.reordered, 0U);
}

TEST(SparseAwareScheduler, leaves_unlabelled_traffic_on_baseline_path)
{
    auto memory = std::make_shared<MultiChannelMemorySystem>(
        "ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm.ini", ".", "unlabelled", 256 * 64);
    memory->setSparseSchedulingEnabled(true);
    BurstType data;
    for (uint64_t address = 0; address < 4096; address += 32)
        ASSERT_TRUE(memory->addTransaction(false, address, &data));
    while (memory->hasPendingTransactions())
        memory->update();

    for (auto* channel : memory->channels)
    {
        const auto& stats = channel->memoryController->getSparseSchedulerStats();
        EXPECT_EQ(stats.sparseTransactions, 0U);
        EXPECT_EQ(stats.reorderedTransactions, 0U);
    }
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

#include <memory>
#include <vector>

#include "MultiChannelMemorySystem.h"
#include "SparseExecution.h"
#include "SparseMapper.h"
#include "SparseMatrix.h"
#include "gtest/gtest.h"

using namespace DRAMSim;

TEST(SparseMatrixTest, NormalizesDuplicatesZerosAndEmptyRows)
{
    std::vector<SparseCOOEntry> entries{{0, 3, 2.0f}, {0, 1, 1.0f}, {0, 1, 4.0f},
                                        {1, 2, 0.0f}, {2, 0, -1.0f}};
    SparseMatrixCSR matrix = SparseMatrixCSR::fromCOO(4, 5, entries);

    ASSERT_EQ(matrix.rowPtr, (std::vector<uint32_t>{0, 2, 2, 3, 3}));
    ASSERT_EQ(matrix.columnIndices, (std::vector<uint32_t>{1, 3, 0}));
    EXPECT_FLOAT_EQ(convertH2F(matrix.values[0]), 5.0f);
    EXPECT_EQ(matrix.rowNnz(1), 0u);
    EXPECT_EQ(matrix.rowNnz(3), 0u);
}

TEST(SparseMatrixTest, PacksMasksAtBurstBoundaries)
{
    SparseMatrixCSR matrix = SparseMatrixCSR::makeSynthetic({0, 1, 15, 16, 17, 127, 128, 129},
                                                            160, 7);
    std::vector<SparseSegment> segments = matrix.packSegments();

    EXPECT_EQ(matrix.rowSegments(0), 0u);
    EXPECT_EQ(matrix.rowSegments(1), 1u);
    EXPECT_EQ(matrix.rowSegments(3), 1u);
    EXPECT_EQ(matrix.rowSegments(4), 2u);
    EXPECT_EQ(matrix.rowSegments(6), 8u);
    EXPECT_EQ(matrix.rowSegments(7), 9u);

    uint32_t row4Offset = matrix.rowSegments(0) + matrix.rowSegments(1) + matrix.rowSegments(2) +
                          matrix.rowSegments(3);
    EXPECT_EQ(segments[row4Offset].validLaneMask, 0xffffu);
    EXPECT_EQ(segments[row4Offset + 1].validLaneMask, 0x0001u);
    EXPECT_EQ(segments[row4Offset + 1].validLanes(), 1u);
}

TEST(SparseMatrixTest, LoadsAndExpandsSymmetricMatrixMarketInput)
{
    SparseMatrixCSR matrix =
        SparseMatrixCSR::loadMatrixMarket("data/sparse/test_symmetric.mtx");
    EXPECT_EQ(matrix.numRows, 3u);
    EXPECT_EQ(matrix.numCols, 3u);
    EXPECT_EQ(matrix.nnz(), 6u);
    EXPECT_EQ(matrix.rowPtr, (std::vector<uint32_t>{0, 2, 5, 6}));
    EXPECT_EQ(matrix.columnIndices, (std::vector<uint32_t>{0, 1, 0, 1, 2, 1}));
}

TEST(SparseMapperTest, GreedyBalancesSkewedRowsAndKeepsRowsAtomic)
{
    SparseMatrixCSR matrix =
        SparseMatrixCSR::makeSynthetic({128, 112, 96, 80, 32, 16, 16, 16, 16, 16}, 160, 9);
    SparseBankMapper mapper(1, 1, 4);
    SparseMappingPlan contiguous = mapper.map(matrix, SparseMappingPolicy::CONTIGUOUS);
    SparseMappingPlan greedy = mapper.map(matrix, SparseMappingPolicy::BANK_AWARE);
    SparseMappingStats contiguousStats = mapper.measure(matrix, contiguous);
    SparseMappingStats greedyStats = mapper.measure(matrix, greedy);

    EXPECT_LE(greedyStats.maxToMeanImbalance, contiguousStats.maxToMeanImbalance);
    std::vector<uint32_t> appearances(matrix.numRows, 0);
    for (const SparseWorkerQueue& worker : greedy.workers)
        for (uint32_t row : worker.rowIds) appearances[row]++;
    for (uint32_t row = 0; row < matrix.numRows; ++row)
        EXPECT_EQ(appearances[row], matrix.rowNnz(row) == 0 ? 0u : 1u);
}

TEST(SparseMapperTest, AllZeroMatrixProducesEmptyWorkerQueues)
{
    SparseMatrixCSR matrix = SparseMatrixCSR::makeSynthetic({0, 0, 0, 0}, 8, 2);
    SparseBankMapper mapper(1, 1, 8);
    SparseMappingPlan plan = mapper.map(matrix, SparseMappingPolicy::BANK_AWARE);
    SparseMappingStats stats = mapper.measure(matrix, plan);
    EXPECT_EQ(stats.totalRows, 0u);
    EXPECT_EQ(stats.totalNnz, 0u);
    EXPECT_EQ(stats.totalSegments, 0u);
    for (const SparseWorkerQueue& worker : plan.workers) EXPECT_TRUE(worker.rowIds.empty());
}

#ifndef NO_STORAGE
TEST(SparseExecutionTest, HostGatherLockstepMatchesFp16Reference)
{
    SparseMatrixCSR matrix =
        SparseMatrixCSR::makeSynthetic({129, 16, 17, 31, 4, 0, 9, 20, 128, 1}, 160, 4);
    std::vector<fp16> input(matrix.numCols);
    for (uint32_t index = 0; index < matrix.numCols; ++index)
        input[index] = convertF2H(static_cast<float>(static_cast<int>(index % 11) - 5) / 8.0f);

    auto memory = std::make_shared<MultiChannelMemorySystem>(
        "ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm_1ch.ini", ".", "sparse_test", 256);
    SparseBankMapper mapper(1, 1, 8);
    SparseMappingPlan plan = mapper.map(matrix, SparseMappingPolicy::BANK_AWARE);
    HostGatherLockstepExecutor executor(memory, 1, 1);
    SparseExecutionResult actual = executor.run(matrix, plan, input);
    std::vector<fp16> expected = matrix.multiplyReference(input);

    ASSERT_EQ(actual.values.size(), expected.size());
    for (uint32_t row = 0; row < matrix.numRows; ++row)
        EXPECT_TRUE(fp16Equal(actual.values[row], expected[row], 32, 0.03f))
            << "row " << row << ": expected " << convertH2F(expected[row]) << ", got "
            << convertH2F(actual.values[row]);
    EXPECT_GT(actual.stats.cycles.pimCompute, 0u);
    EXPECT_GT(actual.stats.reloadRounds, 0u);
    EXPECT_EQ(actual.stats.usefulSegments + actual.stats.wavePaddingSegments,
              actual.stats.scheduledSegments);
}
#endif

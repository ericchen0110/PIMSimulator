#ifndef PIMSIM_SPARSE_EXECUTION_H
#define PIMSIM_SPARSE_EXECUTION_H

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "MultiChannelMemorySystem.h"
#include "PIMCmd.h"
#include "SparseMapper.h"
#include "tests/KernelAddrGen.h"

namespace DRAMSim
{

enum class SparsePhase
{
    MATRIX_SETUP,
    VECTOR_STAGING,
    PIM_COMPUTE,
    RESULT_READBACK
};

const char* requestClassName(RequestClass requestClass);
const char* sparsePhaseName(SparsePhase phase);

struct SparseTrafficCounter
{
    struct ByClass
    {
        uint64_t transactions = 0;
        uint64_t reads = 0;
        uint64_t writes = 0;
        uint64_t bytes = 0;
        uint64_t effectiveBankBursts = 0;
    };

    uint64_t transactions;
    uint64_t reads;
    uint64_t writes;
    uint64_t bytes;
    uint64_t effectiveBankBursts;
    std::map<RequestClass, ByClass> byClass;

    SparseTrafficCounter();
};

struct SparsePhaseCycles
{
    uint64_t matrixSetup;
    uint64_t vectorStaging;
    uint64_t pimCompute;
    uint64_t resultReadback;
    uint64_t recurringSpmv;
    uint64_t endToEnd;

    SparsePhaseCycles();
};

struct SparseLockstepWave
{
    uint32_t channel;
    uint32_t rank;
    uint32_t waveIndex;
    uint32_t segmentCount;
    uint32_t dataBaseRow;
    uint32_t resultRow;
    uint64_t storageOffset;
    std::vector<int64_t> rowByBlock;
};

struct SparseExecutionLayout
{
    std::vector<SparseLockstepWave> waves;
    std::vector<BurstType> weightBursts;
    std::vector<BurstType> vectorBursts;
    std::vector<BurstType> resultBursts;
    std::vector<BurstType> crfBursts;
};

struct SparseRunStats
{
    uint64_t usefulSegments;
    uint64_t scheduledSegments;
    uint64_t wavePaddingSegments;
    uint64_t usefulVectorBursts;
    uint64_t paddedVectorBursts;
    uint64_t grfOverwriteBursts;
    uint64_t reloadRounds;
    uint64_t cpuIndexBytes;
    uint64_t cpuGatherValueBytes;
    std::vector<uint64_t> workerScheduledSegments;
    std::vector<uint64_t> workerEffectiveBankBursts;
    SparsePhaseCycles cycles;
    std::map<SparsePhase, SparseTrafficCounter> traffic;
    MemoryStatsSnapshot memoryBefore;
    MemoryStatsSnapshot memoryAfter;

    SparseRunStats();
};

struct SparseExecutionResult
{
    std::vector<fp16> values;
    SparseRunStats stats;
};

class HostGatherLockstepExecutor
{
  public:
    HostGatherLockstepExecutor(std::shared_ptr<MultiChannelMemorySystem> memory,
                               uint32_t pimChannels, uint32_t pimRanks);

    SparseExecutionLayout buildLayout(const SparseMatrixCSR& matrix,
                                      const SparseMappingPlan& plan) const;
    SparseExecutionResult run(const SparseMatrixCSR& matrix, const SparseMappingPlan& plan,
                              const std::vector<fp16>& input);

  private:
    void queueMatrixSetup(const SparseMatrixCSR& matrix, const SparseMappingPlan& plan,
                          SparseExecutionLayout& layout, SparseRunStats& stats);
    void queueVectorStaging(const SparseMappingPlan& plan, const std::vector<fp16>& input,
                            SparseExecutionLayout& layout, SparseRunStats& stats);
    void queueCompute(SparseExecutionLayout& layout, SparseRunStats& stats);
    void queueReadback(const SparseExecutionLayout& layout, SparseExecutionLayout& mutableLayout,
                       SparseRunStats& stats);

    uint64_t address(uint32_t channel, uint32_t rank, uint32_t flatBank, uint32_t row,
                     uint32_t column) const;
    uint64_t pimAddress(uint32_t channel, uint32_t rank, uint32_t bankParity, uint32_t row,
                        uint32_t column) const;
    void submit(bool isWrite, uint64_t address, const std::string& tag, BurstType* data,
                RequestClass requestClass, SparsePhase phase, SparseRunStats& stats,
                uint32_t effectiveBankBursts = 1);
    void barrier(uint32_t channel);
    uint64_t drain();
    void parkIn(uint32_t channel, uint32_t rank, SparseRunStats& stats);
    void parkOut(uint32_t channel, uint32_t rank, SparseRunStats& stats);
    void enterHab(uint32_t channel, uint32_t rank, SparseRunStats& stats);
    void exitHab(uint32_t channel, uint32_t rank, SparseRunStats& stats);
    void programWave(uint32_t channel, uint32_t rank, SparseLockstepWave& wave,
                     BurstType& crfBurst, SparseRunStats& stats);

    std::shared_ptr<MultiChannelMemorySystem> memory_;
    std::shared_ptr<PIMAddrManager> addressManager_;
    uint32_t channels_;
    uint32_t ranks_;
    uint32_t banks_;
    uint32_t bankGroups_;
    uint32_t pimBlocks_;
    uint32_t accessibleRows_;
    uint32_t columnsPerBurstRow_;
    BurstType nullBurst_;
    BurstType controlOn_;
    BurstType controlOff_;
};

}  // namespace DRAMSim

#endif

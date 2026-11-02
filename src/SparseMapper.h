#ifndef PIMSIM_SPARSE_MAPPER_H
#define PIMSIM_SPARSE_MAPPER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "SparseMatrix.h"

namespace DRAMSim
{

enum class SparseMappingPolicy
{
    CONTIGUOUS,
    ROUND_ROBIN,
    BANK_AWARE
};

const char* sparseMappingPolicyName(SparseMappingPolicy policy);
SparseMappingPolicy parseSparseMappingPolicy(const std::string& value);

struct PIMWorkerId
{
    uint32_t channel;
    uint32_t rank;
    uint32_t pimBlock;

    bool operator==(const PIMWorkerId& rhs) const;
};

struct SparseWorkerQueue
{
    PIMWorkerId worker;
    std::vector<uint32_t> rowIds;
    uint64_t estimatedCost;
    uint64_t usefulSegments;
    uint64_t usefulNnz;

    SparseWorkerQueue();
};

struct SparseMappingPlan
{
    SparseMappingPolicy policy;
    uint32_t numRows;
    uint32_t numCols;
    std::vector<SparseSegment> segments;
    std::vector<uint32_t> rowSegmentOffsets;
    std::vector<SparseWorkerQueue> workers;
    std::vector<uint32_t> outputPermutation;

    const SparseSegment& segment(uint32_t row, uint32_t segmentId) const;
};

class SparseCostModel
{
  public:
    virtual ~SparseCostModel() {}
    virtual uint64_t estimate(uint32_t nnz, uint32_t segments) const = 0;
};

class HostGatherCostModel : public SparseCostModel
{
  public:
    uint64_t estimate(uint32_t nnz, uint32_t segments) const override;
};

struct SparseWorkerStats
{
    PIMWorkerId worker;
    uint64_t rows;
    uint64_t nnz;
    uint64_t segments;
    uint64_t estimatedCost;
};

struct SparseMappingStats
{
    std::vector<SparseWorkerStats> workers;
    uint64_t totalRows;
    uint64_t totalNnz;
    uint64_t totalSegments;
    uint64_t allocatedLanes;
    uint64_t lanePadding;
    uint64_t totalEstimatedCost;
    uint64_t idealMaxEstimatedCost;
    double meanEstimatedCost;
    double maxToMeanImbalance;
    double coefficientOfVariation;
};

class SparseBankMapper
{
  public:
    SparseBankMapper(uint32_t channels, uint32_t ranks, uint32_t pimBlocks,
                     std::shared_ptr<const SparseCostModel> costModel =
                         std::shared_ptr<const SparseCostModel>(new HostGatherCostModel()));

    SparseMappingPlan map(const SparseMatrixCSR& matrix, SparseMappingPolicy policy) const;
    SparseMappingStats measure(const SparseMatrixCSR& matrix,
                               const SparseMappingPlan& plan) const;

  private:
    uint32_t channels_;
    uint32_t ranks_;
    uint32_t pimBlocks_;
    std::shared_ptr<const SparseCostModel> costModel_;
};

}  // namespace DRAMSim

#endif

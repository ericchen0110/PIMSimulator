#include "SparseMapper.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace DRAMSim
{

namespace
{
struct RowWork
{
    uint32_t row;
    uint32_t nnz;
    uint32_t segments;
    uint64_t cost;
};

}  // namespace

const char* sparseMappingPolicyName(SparseMappingPolicy policy)
{
    switch (policy)
    {
        case SparseMappingPolicy::CONTIGUOUS:
            return "contiguous";
        case SparseMappingPolicy::ROUND_ROBIN:
            return "round-robin";
        case SparseMappingPolicy::BANK_AWARE:
            return "bank-aware";
    }
    return "unknown";
}

SparseMappingPolicy parseSparseMappingPolicy(const std::string& value)
{
    if (value == "contiguous")
        return SparseMappingPolicy::CONTIGUOUS;
    if (value == "round-robin" || value == "round_robin")
        return SparseMappingPolicy::ROUND_ROBIN;
    if (value == "bank-aware" || value == "bank_aware")
        return SparseMappingPolicy::BANK_AWARE;
    throw std::invalid_argument("unknown sparse mapping policy: " + value);
}

bool PIMWorkerId::operator==(const PIMWorkerId& rhs) const
{
    return channel == rhs.channel && rank == rhs.rank && pimBlock == rhs.pimBlock;
}

SparseWorkerQueue::SparseWorkerQueue()
    : worker{0, 0, 0}, estimatedCost(0), usefulSegments(0), usefulNnz(0)
{
}

const SparseSegment& SparseMappingPlan::segment(uint32_t row, uint32_t segmentId) const
{
    if (row >= numRows || rowSegmentOffsets.size() != static_cast<size_t>(numRows) + 1)
        throw std::out_of_range("sparse mapping row is outside the plan");
    uint32_t index = rowSegmentOffsets[row] + segmentId;
    if (index >= rowSegmentOffsets[row + 1])
        throw std::out_of_range("sparse segment is outside the mapped row");
    return segments[index];
}

uint64_t HostGatherCostModel::estimate(uint32_t nnz, uint32_t segments) const
{
    return nnz == 0 ? 0 : static_cast<uint64_t>(3) * segments + 1;
}

SparseBankMapper::SparseBankMapper(uint32_t channels, uint32_t ranks, uint32_t pimBlocks,
                                   std::shared_ptr<const SparseCostModel> costModel)
    : channels_(channels), ranks_(ranks), pimBlocks_(pimBlocks), costModel_(costModel)
{
    if (channels == 0 || ranks == 0 || pimBlocks == 0)
        throw std::invalid_argument("sparse mapping topology dimensions must be nonzero");
    if (!costModel_)
        throw std::invalid_argument("sparse mapper needs a cost model");
}

SparseMappingPlan SparseBankMapper::map(const SparseMatrixCSR& matrix,
                                        SparseMappingPolicy policy) const
{
    matrix.validate();
    SparseMappingPlan plan;
    plan.policy = policy;
    plan.numRows = matrix.numRows;
    plan.numCols = matrix.numCols;
    plan.segments = matrix.packSegments();
    plan.rowSegmentOffsets.resize(static_cast<size_t>(matrix.numRows) + 1, 0);
    for (uint32_t row = 0; row < matrix.numRows; ++row)
        plan.rowSegmentOffsets[row + 1] = plan.rowSegmentOffsets[row] + matrix.rowSegments(row);
    plan.outputPermutation.resize(matrix.numRows);
    for (uint32_t row = 0; row < matrix.numRows; ++row) plan.outputPermutation[row] = row;

    uint32_t workerCount = channels_ * ranks_ * pimBlocks_;
    plan.workers.resize(workerCount);
    for (uint32_t channel = 0; channel < channels_; ++channel)
        for (uint32_t rank = 0; rank < ranks_; ++rank)
            for (uint32_t block = 0; block < pimBlocks_; ++block)
            {
                uint32_t index = (channel * ranks_ + rank) * pimBlocks_ + block;
                plan.workers[index].worker = {channel, rank, block};
            }

    std::vector<RowWork> rows;
    rows.reserve(matrix.numRows);
    for (uint32_t row = 0; row < matrix.numRows; ++row)
    {
        uint32_t nnz = matrix.rowNnz(row);
        if (nnz == 0)
            continue;
        uint32_t segments = matrix.rowSegments(row);
        rows.push_back({row, nnz, segments, costModel_->estimate(nnz, segments)});
    }

    if (policy == SparseMappingPolicy::BANK_AWARE)
    {
        std::sort(rows.begin(), rows.end(), [](const RowWork& lhs, const RowWork& rhs) {
            if (lhs.cost != rhs.cost)
                return lhs.cost > rhs.cost;
            if (lhs.segments != rhs.segments)
                return lhs.segments > rhs.segments;
            if (lhs.nnz != rhs.nnz)
                return lhs.nnz > rhs.nnz;
            return lhs.row < rhs.row;
        });
    }

    for (const RowWork& row : rows)
    {
        uint32_t selected = 0;
        if (policy == SparseMappingPolicy::CONTIGUOUS)
        {
            selected = matrix.numRows == 0
                           ? 0
                           : std::min<uint32_t>((static_cast<uint64_t>(row.row) * workerCount) /
                                                    matrix.numRows,
                                                workerCount - 1);
        }
        else if (policy == SparseMappingPolicy::ROUND_ROBIN)
        {
            selected = row.row % workerCount;
        }
        else
        {
            for (uint32_t candidate = 1; candidate < workerCount; ++candidate)
            {
                const SparseWorkerQueue& best = plan.workers[selected];
                const SparseWorkerQueue& current = plan.workers[candidate];
                bool prefer = current.estimatedCost < best.estimatedCost;
                if (current.estimatedCost == best.estimatedCost)
                    prefer = current.usefulSegments < best.usefulSegments;
                if (current.estimatedCost == best.estimatedCost &&
                    current.usefulSegments == best.usefulSegments)
                    prefer = current.rowIds.size() < best.rowIds.size();
                if (prefer)
                    selected = candidate;
            }
        }

        SparseWorkerQueue& worker = plan.workers[selected];
        worker.rowIds.push_back(row.row);
        worker.estimatedCost += row.cost;
        worker.usefulSegments += row.segments;
        worker.usefulNnz += row.nnz;
    }

    for (SparseWorkerQueue& worker : plan.workers)
    {
        std::stable_sort(worker.rowIds.begin(), worker.rowIds.end(), [&](uint32_t lhs, uint32_t rhs) {
            uint32_t lhsSegments = matrix.rowSegments(lhs);
            uint32_t rhsSegments = matrix.rowSegments(rhs);
            if (lhsSegments != rhsSegments)
                return lhsSegments > rhsSegments;
            return lhs < rhs;
        });
    }
    return plan;
}

SparseMappingStats SparseBankMapper::measure(const SparseMatrixCSR& matrix,
                                             const SparseMappingPlan& plan) const
{
    matrix.validate();
    if (plan.numRows != matrix.numRows || plan.numCols != matrix.numCols)
        throw std::invalid_argument("mapping plan dimensions do not match the sparse matrix");

    SparseMappingStats stats{};
    stats.workers.reserve(plan.workers.size());
    uint64_t maxCost = 0;
    for (const SparseWorkerQueue& worker : plan.workers)
    {
        SparseWorkerStats workerStats{worker.worker, worker.rowIds.size(), worker.usefulNnz,
                                      worker.usefulSegments, worker.estimatedCost};
        stats.workers.push_back(workerStats);
        stats.totalRows += workerStats.rows;
        stats.totalNnz += workerStats.nnz;
        stats.totalSegments += workerStats.segments;
        stats.totalEstimatedCost += workerStats.estimatedCost;
        maxCost = std::max(maxCost, workerStats.estimatedCost);
    }
    stats.allocatedLanes = stats.totalSegments * 16;
    stats.lanePadding = stats.allocatedLanes - stats.totalNnz;
    if (!stats.workers.empty())
    {
        stats.meanEstimatedCost = static_cast<double>(stats.totalEstimatedCost) /
                                  static_cast<double>(stats.workers.size());
        stats.idealMaxEstimatedCost =
            (stats.totalEstimatedCost + stats.workers.size() - 1) / stats.workers.size();
        if (stats.meanEstimatedCost > 0.0)
        {
            stats.maxToMeanImbalance = static_cast<double>(maxCost) / stats.meanEstimatedCost;
            double squaredDeviation = 0.0;
            for (const SparseWorkerStats& worker : stats.workers)
            {
                double delta = worker.estimatedCost - stats.meanEstimatedCost;
                squaredDeviation += delta * delta;
            }
            squaredDeviation /= stats.workers.size();
            stats.coefficientOfVariation = std::sqrt(squaredDeviation) / stats.meanEstimatedCost;
        }
    }
    return stats;
}

}  // namespace DRAMSim

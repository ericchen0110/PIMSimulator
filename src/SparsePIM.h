#ifndef SPARSE_PIM_H
#define SPARSE_PIM_H

#include <cstdint>
#include <functional>
#include <vector>
#include "Burst.h"

namespace DRAMSim
{
class MultiChannelMemorySystem;
enum class SparsePIMPolicy { Lockstep, Independent };

// CSR arrays and a replicated FP32 input vector reside in each block's bank pair.
// Offsets are bytes in that bank pair, not host pointers. Rows may be assigned by
// any static mapping; runtime traversal and scheduling are entirely device-owned.
struct SparsePIMPartition
{
    unsigned channel = 0, rank = 0, block = 0;
    uint32_t rows = 0, entries = 0;
    uint64_t rowPtr = 0, columns = 0, values = 0, input = 0, output = 0;
};
struct SparsePIMJob
{
    uint32_t columns = 0;
    std::vector<SparsePIMPartition> partitions;
    // Optional placement adapter, shared by loading and device address generation.
    std::function<uint64_t(const SparsePIMPartition&, uint64_t)> address;
};

class SparsePIM
{
  public:
    enum class PC { RowStart, RowEnd, Column, Value, Input, Mac, EntryBarrier,
                    RowBarrier, Store, AdvanceRow, Done };
    struct BlockState
    {
        PC pc = PC::RowStart;
        uint32_t row = 0, cursor = 0, end = 0, column = 0;
        float value = 0, input = 0, accumulator = 0;
        bool waiting = false;
        uint64_t readyCycle = 0, completedRows = 0, macs = 0;
        BurstType buffer;
    };
    struct Stats
    {
        uint64_t cycles = 0, macs = 0, completedRows = 0, instructions = 0;
        uint64_t memoryWaitCycles = 0, barrierCycles = 0, maxRowLead = 0;
    };

    SparsePIM(MultiChannelMemorySystem&, const SparsePIMJob&, SparsePIMPolicy, unsigned);
    void update();
    bool busy() const;
    const Stats& stats() const { return stats_; }
    const std::vector<BlockState>& blocks() const { return blocks_; }
    static uint64_t bankAddress(const SparsePIMPartition&, uint64_t byteOffset);

  private:
    MultiChannelMemorySystem& memory;
    SparsePIMJob job;
    SparsePIMPolicy policy;
    unsigned macLatency;
    uint64_t cycle = 0;
    Stats stats_;
    std::vector<BlockState> blocks_;
    std::vector<std::vector<size_t>> groups;
    void request(size_t block, uint64_t offset, bool write, PC next);
    void advance(size_t block);
    void releaseBarriers();
};
} // namespace DRAMSim
#endif

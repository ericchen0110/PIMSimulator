#ifndef PIMSIM_SPARSE_MATRIX_H
#define PIMSIM_SPARSE_MATRIX_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "Burst.h"

namespace DRAMSim
{

struct SparseCOOEntry
{
    uint32_t row;
    uint32_t column;
    float value;
};

struct SparseSegment
{
    uint32_t rowId;
    uint32_t segmentId;
    uint16_t validLaneMask;
    BurstType values;
    std::array<uint32_t, 16> columnIndices;

    SparseSegment();
    uint32_t validLanes() const;
};

struct SparseMatrixCSR
{
    uint32_t numRows;
    uint32_t numCols;
    std::vector<uint32_t> rowPtr;
    std::vector<uint32_t> columnIndices;
    std::vector<fp16> values;

    SparseMatrixCSR();

    static SparseMatrixCSR fromCOO(uint32_t rows, uint32_t columns,
                                   const std::vector<SparseCOOEntry>& entries);
    static SparseMatrixCSR loadMatrixMarket(const std::string& path);
    static SparseMatrixCSR makeSynthetic(const std::vector<uint32_t>& rowNonzeros,
                                         uint32_t columns, uint32_t seed = 1);

    void validate() const;
    uint32_t nnz() const;
    uint32_t rowNnz(uint32_t row) const;
    uint32_t rowSegments(uint32_t row) const;
    std::vector<SparseSegment> packSegments() const;
    std::vector<fp16> multiplyReference(const std::vector<fp16>& input) const;
};

}  // namespace DRAMSim

#endif

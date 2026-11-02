#include "SparseMatrix.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>

namespace DRAMSim
{

namespace
{
uint32_t popcount16(uint16_t value)
{
    uint32_t count = 0;
    while (value != 0)
    {
        count += value & 1u;
        value >>= 1;
    }
    return count;
}

std::string lowercase(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}
}  // namespace

SparseSegment::SparseSegment() : rowId(0), segmentId(0), validLaneMask(0)
{
    columnIndices.fill(0);
}

uint32_t SparseSegment::validLanes() const
{
    return popcount16(validLaneMask);
}

SparseMatrixCSR::SparseMatrixCSR() : numRows(0), numCols(0) {}

SparseMatrixCSR SparseMatrixCSR::fromCOO(uint32_t rows, uint32_t columns,
                                         const std::vector<SparseCOOEntry>& entries)
{
    SparseMatrixCSR result;
    result.numRows = rows;
    result.numCols = columns;

    std::vector<std::map<uint32_t, float>> normalized(rows);
    for (const SparseCOOEntry& entry : entries)
    {
        if (entry.row >= rows || entry.column >= columns)
            throw std::invalid_argument("sparse COO coordinate is outside matrix dimensions");
        if (entry.value != 0.0f)
            normalized[entry.row][entry.column] += entry.value;
    }

    result.rowPtr.reserve(static_cast<size_t>(rows) + 1);
    result.rowPtr.push_back(0);
    for (uint32_t row = 0; row < rows; ++row)
    {
        for (const auto& item : normalized[row])
        {
            if (item.second == 0.0f)
                continue;
            result.columnIndices.push_back(item.first);
            result.values.push_back(convertF2H(item.second));
        }
        result.rowPtr.push_back(static_cast<uint32_t>(result.values.size()));
    }

    result.validate();
    return result;
}

SparseMatrixCSR SparseMatrixCSR::loadMatrixMarket(const std::string& path)
{
    std::ifstream input(path.c_str());
    if (!input)
        throw std::runtime_error("cannot open Matrix Market file: " + path);

    std::string header;
    if (!std::getline(input, header))
        throw std::runtime_error("empty Matrix Market file: " + path);

    std::istringstream headerStream(header);
    std::string banner, object, format, field, symmetry;
    headerStream >> banner >> object >> format >> field >> symmetry;
    banner = lowercase(banner);
    object = lowercase(object);
    format = lowercase(format);
    field = lowercase(field);
    symmetry = lowercase(symmetry);

    if (banner != "%%matrixmarket" || object != "matrix" || format != "coordinate")
        throw std::runtime_error("only Matrix Market coordinate matrices are supported");
    if (field != "real" && field != "integer" && field != "pattern")
        throw std::runtime_error("unsupported Matrix Market value field: " + field);
    if (symmetry != "general" && symmetry != "symmetric")
        throw std::runtime_error("unsupported Matrix Market symmetry: " + symmetry);

    std::string line;
    do
    {
        if (!std::getline(input, line))
            throw std::runtime_error("Matrix Market size line is missing");
    } while (line.empty() || line[0] == '%');

    uint32_t rows = 0, columns = 0;
    uint64_t declaredNnz = 0;
    std::istringstream sizeStream(line);
    if (!(sizeStream >> rows >> columns >> declaredNnz))
        throw std::runtime_error("invalid Matrix Market size line");

    std::vector<SparseCOOEntry> entries;
    entries.reserve(static_cast<size_t>(declaredNnz) * (symmetry == "symmetric" ? 2 : 1));
    uint64_t parsed = 0;
    while (parsed < declaredNnz && std::getline(input, line))
    {
        if (line.empty() || line[0] == '%')
            continue;
        std::istringstream entryStream(line);
        uint32_t rowOneBased = 0, colOneBased = 0;
        float value = 1.0f;
        if (!(entryStream >> rowOneBased >> colOneBased))
            throw std::runtime_error("invalid Matrix Market coordinate");
        if (field != "pattern" && !(entryStream >> value))
            throw std::runtime_error("Matrix Market value is missing");
        if (rowOneBased == 0 || colOneBased == 0)
            throw std::runtime_error("Matrix Market coordinates must be one-based");

        uint32_t row = rowOneBased - 1;
        uint32_t column = colOneBased - 1;
        entries.push_back({row, column, value});
        if (symmetry == "symmetric" && row != column)
            entries.push_back({column, row, value});
        ++parsed;
    }

    if (parsed != declaredNnz)
        throw std::runtime_error("Matrix Market entry count does not match its header");
    return fromCOO(rows, columns, entries);
}

SparseMatrixCSR SparseMatrixCSR::makeSynthetic(const std::vector<uint32_t>& rowNonzeros,
                                               uint32_t columns, uint32_t seed)
{
    if (columns == 0 && !rowNonzeros.empty())
        throw std::invalid_argument("synthetic sparse matrix needs at least one column");

    std::mt19937 generator(seed);
    std::vector<SparseCOOEntry> entries;
    for (uint32_t row = 0; row < rowNonzeros.size(); ++row)
    {
        if (rowNonzeros[row] > columns)
            throw std::invalid_argument("synthetic row NNZ exceeds the number of columns");

        std::vector<uint32_t> candidates(columns);
        for (uint32_t column = 0; column < columns; ++column) candidates[column] = column;
        std::shuffle(candidates.begin(), candidates.end(), generator);
        candidates.resize(rowNonzeros[row]);
        std::sort(candidates.begin(), candidates.end());

        for (uint32_t column : candidates)
        {
            float value = 0.25f + static_cast<float>((row + column) % 13) / 8.0f;
            entries.push_back({row, column, value});
        }
    }
    return fromCOO(static_cast<uint32_t>(rowNonzeros.size()), columns, entries);
}

void SparseMatrixCSR::validate() const
{
    if (rowPtr.size() != static_cast<size_t>(numRows) + 1)
        throw std::invalid_argument("CSR rowPtr size must be numRows + 1");
    if (rowPtr.empty() || rowPtr.front() != 0)
        throw std::invalid_argument("CSR rowPtr must begin with zero");
    if (columnIndices.size() != values.size() || rowPtr.back() != values.size())
        throw std::invalid_argument("CSR value, column, and rowPtr sizes are inconsistent");

    for (uint32_t row = 0; row < numRows; ++row)
    {
        if (rowPtr[row] > rowPtr[row + 1])
            throw std::invalid_argument("CSR rowPtr must be nondecreasing");
        uint32_t previous = 0;
        bool first = true;
        for (uint32_t index = rowPtr[row]; index < rowPtr[row + 1]; ++index)
        {
            if (columnIndices[index] >= numCols)
                throw std::invalid_argument("CSR column index is outside matrix dimensions");
            if (!first && columnIndices[index] <= previous)
                throw std::invalid_argument("CSR columns must be sorted and unique per row");
            previous = columnIndices[index];
            first = false;
        }
    }
}

uint32_t SparseMatrixCSR::nnz() const
{
    return static_cast<uint32_t>(values.size());
}

uint32_t SparseMatrixCSR::rowNnz(uint32_t row) const
{
    if (row >= numRows)
        throw std::out_of_range("sparse row is outside matrix dimensions");
    return rowPtr[row + 1] - rowPtr[row];
}

uint32_t SparseMatrixCSR::rowSegments(uint32_t row) const
{
    return (rowNnz(row) + 15u) / 16u;
}

std::vector<SparseSegment> SparseMatrixCSR::packSegments() const
{
    validate();
    std::vector<SparseSegment> packed;
    packed.reserve((values.size() + 15) / 16);

    for (uint32_t row = 0; row < numRows; ++row)
    {
        uint32_t begin = rowPtr[row];
        uint32_t end = rowPtr[row + 1];
        uint32_t segmentId = 0;
        for (uint32_t offset = begin; offset < end; offset += 16, ++segmentId)
        {
            SparseSegment segment;
            segment.rowId = row;
            segment.segmentId = segmentId;
            uint32_t count = std::min<uint32_t>(16, end - offset);
            for (uint32_t lane = 0; lane < count; ++lane)
            {
                segment.values.fp16Data_[lane] = values[offset + lane];
                segment.columnIndices[lane] = columnIndices[offset + lane];
                segment.validLaneMask |= static_cast<uint16_t>(1u << lane);
            }
            packed.push_back(segment);
        }
    }
    return packed;
}

std::vector<fp16> SparseMatrixCSR::multiplyReference(const std::vector<fp16>& input) const
{
    validate();
    if (input.size() != numCols)
        throw std::invalid_argument("SpMV input vector length must equal the matrix column count");

    std::vector<fp16> output(numRows, convertF2H(0.0f));
    std::vector<SparseSegment> segments = packSegments();
    std::vector<std::array<fp16, 16>> laneSums(numRows);
    for (auto& lanes : laneSums)
        for (fp16& lane : lanes) lane = convertF2H(0.0f);

    for (const SparseSegment& segment : segments)
    {
        for (uint32_t lane = 0; lane < 16; ++lane)
        {
            if ((segment.validLaneMask & (1u << lane)) == 0)
                continue;
            laneSums[segment.rowId][lane] = laneSums[segment.rowId][lane] +
                                              segment.values.fp16Data_[lane] *
                                                  input[segment.columnIndices[lane]];
        }
    }
    for (uint32_t row = 0; row < numRows; ++row)
        for (uint32_t lane = 0; lane < 16; ++lane) output[row] += laneSums[row][lane];
    return output;
}

}  // namespace DRAMSim

#ifndef __SPMV_LAYOUT_H__
#define __SPMV_LAYOUT_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "Burst.h"
#include "FP16.h"

using namespace std;
using namespace DRAMSim;

// Sparse matrix in CSR form. Column indices within a row are sorted.
struct CsrMatrix
{
    int rows = 0, cols = 0;
    vector<int> rowPtr;
    vector<int> colIdx;
    vector<float> vals;

    size_t nnz() const
    {
        return colIdx.size();
    }

    // Coefficient of variation of the row lengths (std / mean).
    double rowLengthCv() const
    {
        double mean = (double)nnz() / rows, var = 0;
        for (int r = 0; r < rows; r++)
        {
            double d = (rowPtr[r + 1] - rowPtr[r]) - mean;
            var += d * d;
        }
        return mean > 0 ? sqrt(var / rows) / mean : 0;
    }
};

// Values are multiples of 1/4 so FP16 products and short sums are exact.
inline float spmvTestValue(uint32_t h)
{
    float v = 0.25f * (1 + (h % 4));
    return (h & 4) ? -v : v;
}

inline vector<float> spmvTestVector(int n)
{
    vector<float> x(n);
    for (int j = 0; j < n; j++) x[j] = 0.25f * ((j % 9) - 4);
    return x;
}

inline CsrMatrix makeSparseFromRowLengths(int m, int n, const vector<int>& lengths, mt19937& rng)
{
    CsrMatrix a;
    a.rows = m;
    a.cols = n;
    a.rowPtr.assign(m + 1, 0);
    vector<int> all(n);
    iota(all.begin(), all.end(), 0);
    vector<char> used(n, 0);
    vector<int> picked;
    for (int r = 0; r < m; r++)
    {
        int len = min(lengths[r], n);
        picked.clear();
        if (len * 2 < n)
        {
            // Rejection sampling, then sort. Expected O(len) draws.
            while ((int)picked.size() < len)
            {
                int c = (int)(rng() % n);
                if (!used[c])
                {
                    used[c] = 1;
                    picked.push_back(c);
                }
            }
            for (int c : picked) used[c] = 0;
            sort(picked.begin(), picked.end());
        }
        else
        {
            // Partial Fisher-Yates shuffle (std::sample needs C++17).
            for (int i = 0; i < len; i++) swap(all[i], all[i + rng() % (n - i)]);
            picked.assign(all.begin(), all.begin() + len);
            sort(picked.begin(), picked.end());
        }
        for (int c : picked)
        {
            a.colIdx.push_back(c);
            a.vals.push_back(spmvTestValue(rng()));
        }
        a.rowPtr[r + 1] = (int)a.colIdx.size();
    }
    return a;
}

// Every entry is nonzero with probability `density`.
inline CsrMatrix makeUniformSparse(int m, int n, double density, uint32_t seed)
{
    mt19937 rng(seed);
    binomial_distribution<int> len_dist(n, density);
    vector<int> lengths(m);
    for (int& l : lengths) l = len_dist(rng);
    return makeSparseFromRowLengths(m, n, lengths, rng);
}

// Row lengths follow a Pareto distribution (shape alpha) with the same mean
// as the uniform case, capped at n. Small alpha means a heavier tail.
inline CsrMatrix makePowerLawSparse(int m, int n, double density, double alpha, uint32_t seed)
{
    mt19937 rng(seed);
    uniform_real_distribution<double> u(0.0, 1.0);
    vector<double> w(m);
    for (double& v : w) v = pow(1.0 - u(rng), -1.0 / alpha);
    double target = density * n * m;
    // Rescale so the capped total matches the target number of nonzeros.
    double scale = target / accumulate(w.begin(), w.end(), 0.0);
    vector<int> lengths(m);
    for (int iter = 0; iter < 20; iter++)
    {
        double total = 0;
        for (int r = 0; r < m; r++)
        {
            lengths[r] = (int)min<double>(n, floor(w[r] * scale + u(rng)));
            total += lengths[r];
        }
        if (total <= 0 || fabs(total - target) < 0.005 * target)
            break;
        scale *= target / total;
    }
    return makeSparseFromRowLengths(m, n, lengths, rng);
}

// How a sparse matrix is laid out across PIM channels, blocks, lanes and GRF_B slots.
//
// Each lane of each PIM block owns one matrix row (or one column segment of a row).
// Channels form a grid: rowParts x colParts. Channel ch = p * colParts + q handles
// rows of row partition p and columns of column partition q. The host adds the
// colParts partial sums of each row. colParts is chosen so every channel has about
// 1024 rows (8 blocks x 8 GRF_B slots x 16 lanes), which fills one PIM session.
//
// Within a channel, local row i maps to
//   session s = i / 1024, slot d = (i % 1024) / 128, block b = (i % 128) / 16, lane l = i % 16
// For each 128-column tile of x (one full GRF_A), slot d needs K[d] (LDIDX, SPMAC) pairs,
// where K[d] is the most nonzeros any of its 128 rows has in that tile. All 8 blocks run in
// lockstep, so shorter rows are padded with (idx 0, val 0).
struct SpmvTile
{
    int tile;  // tile index within the channel's column partition
    int k[8];  // pairs per GRF_B slot
};

struct SpmvSession
{
    int numSlots = 0;
    vector<SpmvTile> tiles;  // only tiles with at least one nonzero
};

struct SpmvChannelPlan
{
    int rowBase = 0, numRows = 0, colBase = 0, numCols = 0;
    vector<SpmvSession> sessions;
    uint64_t pairs = 0;  // (LDIDX, SPMAC) pairs issued by this channel
};

class SpmvLayout
{
  public:
    static const int kLanes = 16;
    static const int kSlots = 8;
    static const int kTileCols = 128;

    SpmvLayout(const CsrMatrix& a, int num_chans, int num_blocks, bool keep_data)
        : rows(a.rows), cols(a.cols), numChans(num_chans), numBlocks(num_blocks), nnz(a.nnz())
    {
        int rows_per_session = kSlots * numBlocks * kLanes;
        rowParts = 1;
        while (rowParts * 2 <= numChans && (int64_t)rowParts * 2 * rows_per_session <= rows)
            rowParts *= 2;
        if (numChans % rowParts != 0)
            throw invalid_argument("channel count must be a power of two");
        colParts = numChans / rowParts;

        int rows_per_part = (rows + rowParts - 1) / rowParts;
        int cols_per_part = (cols + colParts - 1) / colParts;
        cols_per_part = (cols_per_part + kTileCols - 1) / kTileCols * kTileCols;

        chans.resize(numChans);
        if (keep_data)
            stream.assign(numChans, vector<vector<BurstType>>(numBlocks));

        for (int ch = 0; ch < numChans; ch++)
        {
            SpmvChannelPlan& cp = chans[ch];
            int p = ch / colParts, q = ch % colParts;
            cp.rowBase = min(rows, p * rows_per_part);
            cp.numRows = min(rows, cp.rowBase + rows_per_part) - cp.rowBase;
            cp.colBase = min(cols, q * cols_per_part);
            cp.numCols = min(cols, cp.colBase + cols_per_part) - cp.colBase;
            planChannel(a, ch, rows_per_session, keep_data);
            paddedPairs += cp.pairs;
        }
    }

    // Local row -> placement inside a channel.
    void place(int local_row, int& session, int& slot, int& block, int& lane) const
    {
        int rows_per_session = kSlots * numBlocks * kLanes;
        session = local_row / rows_per_session;
        int r = local_row % rows_per_session;
        slot = r / (numBlocks * kLanes);
        block = (r % (numBlocks * kLanes)) / kLanes;
        lane = r % kLanes;
    }

    int localRow(int session, int slot, int block, int lane) const
    {
        return ((session * kSlots + slot) * numBlocks + block) * kLanes + lane;
    }

    // Fraction of issued lane-operations that carried a real nonzero.
    double laneFill() const
    {
        return paddedPairs ? (double)nnz / ((double)paddedPairs * numBlocks * kLanes) : 0;
    }

    int rows, cols, numChans, numBlocks;
    int rowParts = 1, colParts = 1;
    uint64_t nnz = 0, paddedPairs = 0;
    vector<SpmvChannelPlan> chans;
    // stream[ch][block] = bursts to preload into that block's even bank, in issue order:
    // idx, val, idx, val, ...
    vector<vector<vector<BurstType>>> stream;

  private:
    void planChannel(const CsrMatrix& a, int ch, int rows_per_session, bool keep_data)
    {
        SpmvChannelPlan& cp = chans[ch];
        if (cp.numRows <= 0 || cp.numCols <= 0)
            return;
        int num_tiles = (cp.numCols + kTileCols - 1) / kTileCols;
        int num_sessions = (cp.numRows + rows_per_session - 1) / rows_per_session;
        int col_end = cp.colBase + cp.numCols;

        // first[i] = CSR position of local row i's first entry in this column partition.
        vector<int> first(cp.numRows), last(cp.numRows);
        for (int i = 0; i < cp.numRows; i++)
        {
            int r = cp.rowBase + i;
            auto b = a.colIdx.begin() + a.rowPtr[r], e = a.colIdx.begin() + a.rowPtr[r + 1];
            first[i] = (int)(lower_bound(b, e, cp.colBase) - a.colIdx.begin());
            last[i] = (int)(lower_bound(b, e, col_end) - a.colIdx.begin());
        }

        cp.sessions.resize(num_sessions);
        vector<int> cursor(first);
        for (int s = 0; s < num_sessions; s++)
        {
            SpmvSession& ss = cp.sessions[s];
            int session_rows = min(rows_per_session, cp.numRows - s * rows_per_session);
            ss.numSlots = (session_rows + numBlocks * kLanes - 1) / (numBlocks * kLanes);

            for (int t = 0; t < num_tiles; t++)
            {
                int tile_end = cp.colBase + (t + 1) * kTileCols;
                SpmvTile tile;
                tile.tile = t;
                int any = 0;
                for (int d = 0; d < kSlots; d++)
                {
                    tile.k[d] = 0;
                    for (int b = 0; b < numBlocks; b++)
                        for (int l = 0; l < kLanes; l++)
                        {
                            int i = localRow(s, d, b, l);
                            if (i >= cp.numRows)
                                continue;
                            int c = cursor[i], n = 0;
                            while (c < last[i] && a.colIdx[c] < tile_end) c++, n++;
                            tile.k[d] = max(tile.k[d], n);
                        }
                    any += tile.k[d];
                }
                if (any == 0)
                    continue;

                for (int d = 0; d < kSlots; d++)
                {
                    cp.pairs += tile.k[d];
                    if (!keep_data)
                        continue;
                    for (int k = 0; k < tile.k[d]; k++)
                        for (int b = 0; b < numBlocks; b++)
                        {
                            BurstType idx, val;
                            for (int l = 0; l < kLanes; l++)
                            {
                                idx.u16Data_[l] = 0;
                                val.fp16Data_[l] = convertF2H(0.0f);
                                int i = localRow(s, d, b, l);
                                if (i >= cp.numRows)
                                    continue;
                                int c = cursor[i] + k;
                                if (c < last[i] && a.colIdx[c] < tile_end)
                                {
                                    idx.u16Data_[l] = a.colIdx[c] - (tile_end - kTileCols);
                                    val.fp16Data_[l] = convertF2H(a.vals[c]);
                                }
                            }
                            stream[ch][b].push_back(idx);
                            stream[ch][b].push_back(val);
                        }
                }
                // Advance every row of this session past the tile.
                for (int d = 0; d < kSlots; d++)
                    for (int b = 0; b < numBlocks; b++)
                        for (int l = 0; l < kLanes; l++)
                        {
                            int i = localRow(s, d, b, l);
                            if (i >= cp.numRows)
                                continue;
                            while (cursor[i] < last[i] && a.colIdx[cursor[i]] < tile_end)
                                cursor[i]++;
                        }
                ss.tiles.push_back(tile);
            }
        }
    }
};

// Reference y = A x in float.
inline vector<float> spmvReference(const CsrMatrix& a, const vector<float>& x)
{
    vector<float> y(a.rows, 0.0f);
    for (int r = 0; r < a.rows; r++)
        for (int c = a.rowPtr[r]; c < a.rowPtr[r + 1]; c++) y[r] += a.vals[c] * x[a.colIdx[c]];
    return y;
}

#endif /* __SPMV_LAYOUT_H__ */

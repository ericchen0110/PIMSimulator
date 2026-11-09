#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <vector>

#include "gtest/gtest.h"
#include "tests/PIMKernel.h"

using namespace DRAMSim;

namespace
{
const char* kDeviceIni = "ini/HBM2_samsung_2M_16B_x64.ini";

shared_ptr<MultiChannelMemorySystem> makeMem(int num_chans)
{
    const char* sys = (num_chans == 1) ? "system_hbm_1ch.ini" : "system_hbm_64ch.ini";
    return make_shared<MultiChannelMemorySystem>(kDeviceIni, sys, ".", "example_app",
                                                 256 * num_chans * 2);
}

// Preloads A, runs SPMAC SpMV, reads y back, and compares with a float reference.
void checkSpmv(const CsrMatrix& a, int num_chans)
{
    auto mem = makeMem(num_chans);
    auto kernel = make_shared<PIMKernel>(mem, num_chans, 1);
    SpmvLayout layout(a, num_chans, getConfigParam(UINT, "NUM_PIM_BLOCKS"), true);
    vector<float> x = spmvTestVector(a.cols);

    kernel->preloadSpmv(layout);
    kernel->runPIM();
    uint64_t before = kernel->getCycle();
    kernel->executeSpmv(layout, x);
    kernel->runPIM();
    uint64_t cycles = kernel->getCycle() - before;
    vector<float> y = kernel->readSpmvResult(layout);

    vector<float> ref = spmvReference(a, x);
    int bad = 0;
    for (int r = 0; r < a.rows; r++)
    {
        if (fabs(y[r] - ref[r]) > 1e-3f * max(1.0f, fabs(ref[r])))
        {
            if (bad++ < 10)
                ADD_FAILURE() << "row " << r << ": pim " << y[r] << " ref " << ref[r];
        }
    }
    EXPECT_EQ(bad, 0);
    cout << "  " << a.rows << "x" << a.cols << " nnz " << a.nnz() << " on " << num_chans
         << " ch: grid " << layout.rowParts << "x" << layout.colParts << ", lane fill "
         << layout.laneFill() << ", " << cycles << " cycles" << endl;
}
}  // namespace

TEST(SpmvKernel, oneChannelMultipleSessions)
{
    // 2300 rows -> 3 sessions on one channel; 300 cols -> 3 tiles, the last one partial.
    checkSpmv(makeUniformSparse(2300, 300, 0.03, 1), 1);
}

TEST(SpmvKernel, oneChannelPowerLawRows)
{
    checkSpmv(makePowerLawSparse(700, 600, 0.02, 1.5, 2), 1);
}

TEST(SpmvKernel, sixtyFourChannelGrid)
{
    // 4 row partitions x 16 column partitions; only 8 column partitions get columns.
    checkSpmv(makeUniformSparse(4096, 1000, 0.01, 3), 64);
}

TEST(SpmvKernel, sixtyFourChannelPowerLaw)
{
    checkSpmv(makePowerLawSparse(2048, 2048, 0.01, 1.5, 4), 64);
}

// ---------------------------------------------------------------------------
// Benchmark: SPMAC SpMV vs dense PIM GEMV (MAC) vs non-PIM host, on 64 channels.
//
// Environment variables (all optional):
//   SPMV_SIZES      e.g. "4096x4096,8192x8192"         (rows x cols)
//   SPMV_DENSITIES  e.g. "0.001,0.01,0.1"
//   SPMV_DISTS      "uniform", "powerlaw", or both
//   SPMV_ALPHA      Pareto shape for powerlaw rows (default 1.5)
//   SPMV_CSV        output file (default spmv_bench.csv)
//
// Like PIMBenchFixture, every number excludes moving the matrix into DRAM: the
// matrix is assumed resident, and only the kernel's own commands are timed.
// ---------------------------------------------------------------------------
namespace
{
vector<string> splitList(const char* env, const string& dflt)
{
    string s = env ? env : dflt;
    vector<string> out;
    stringstream ss(s);
    string item;
    while (getline(ss, item, ','))
        if (!item.empty())
            out.push_back(item);
    return out;
}

uint64_t runToIdle(shared_ptr<MultiChannelMemorySystem> mem)
{
    uint64_t cycle = 0;
    while (mem->hasPendingTransactions())
    {
        mem->update();
        cycle++;
    }
    return cycle;
}

// Same traffic model as PIMBenchTestCase::genMemTraffic: stream bytes linearly.
uint64_t streamTraffic(shared_ptr<MultiChannelMemorySystem> mem, bool is_write, uint64_t bytes,
                       uint64_t start, BurstType* bst)
{
    unsigned stride = getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") * getConfigParam(UINT, "BL") / 8;
    uint64_t addr;
    for (addr = start; addr < start + bytes; addr += stride) mem->addTransaction(is_write, addr, bst);
    return addr;
}

// Host reads inputs, then writes the output, as the existing GEMV bench does.
uint64_t nonPimCycles(uint64_t read_bytes, uint64_t write_bytes)
{
    auto mem = makeMem(64);
    BurstType bst;
    uint64_t addr = streamTraffic(mem, false, read_bytes, 0, &bst);
    uint64_t cycles = runToIdle(mem);
    streamTraffic(mem, true, write_bytes, addr, &bst);
    return cycles + runToIdle(mem);
}

uint64_t denseGemvPimCycles(int m, int n)
{
    auto mem = makeMem(64);
    auto kernel = make_shared<PIMKernel>(mem, 64, 1);
    NumpyBurstType w, in;
    w.bShape = {(unsigned long)m, (unsigned long)(n / 16)};
    int in_tiles = (n / 16 + 7) / 8;
    in.bShape = {1, (unsigned long)(n / 16)};
    in.bData.resize(in_tiles * 8);
    kernel->executeGemv(&w, &in, false);
    kernel->runPIM();
    return kernel->getCycle();
}
}  // namespace

TEST(SpmvBench, sweep)
{
    vector<string> sizes = splitList(getenv("SPMV_SIZES"), "4096x4096");
    vector<string> dens = splitList(getenv("SPMV_DENSITIES"),
                                    "0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5");
    vector<string> dists = splitList(getenv("SPMV_DISTS"), "uniform,powerlaw");
    double alpha = getenv("SPMV_ALPHA") ? atof(getenv("SPMV_ALPHA")) : 1.5;
    string csv_path = getenv("SPMV_CSV") ? getenv("SPMV_CSV") : "spmv_bench.csv";

    ofstream csv(csv_path);
    csv << "rows,cols,dist,density,nnz,row_len_cv,grid,lane_fill,padded_pairs,"
           "spmac_reads,x_writes,crf_writes,pc_writes,flush_writes,"
           "cyc_spmac,cyc_pim_gemv,cyc_host_dense,cyc_host_csr,"
           "speedup_vs_pim_gemv,speedup_vs_host_csr,speedup_vs_host_dense"
        << endl;

    for (const string& size : sizes)
    {
        int m = stoi(size.substr(0, size.find('x')));
        int n = stoi(size.substr(size.find('x') + 1));
        uint64_t cyc_gemv = denseGemvPimCycles(m, n);
        uint64_t cyc_host_dense = nonPimCycles((uint64_t)m * n * 2 + (uint64_t)n * 2, (uint64_t)m * 2);
        cout << "[" << m << "x" << n << "] dense PIM GEMV " << cyc_gemv << " cycles, host dense "
             << cyc_host_dense << " cycles" << endl;

        for (const string& dist : dists)
            for (const string& dstr : dens)
            {
                double d = stod(dstr);
                auto t0 = chrono::steady_clock::now();
                CsrMatrix a = (dist == "powerlaw") ? makePowerLawSparse(m, n, d, alpha, 42)
                                                   : makeUniformSparse(m, n, d, 42);

                auto mem = makeMem(64);
                auto kernel = make_shared<PIMKernel>(mem, 64, 1);
                SpmvLayout layout(a, 64, getConfigParam(UINT, "NUM_PIM_BLOCKS"), false);
                SpmvStats st = kernel->executeSpmv(layout, spmvTestVector(n));
                kernel->runPIM();
                uint64_t cyc_spmac = kernel->getCycle();

                // CSR on the host: FP16 values, 16-bit column indices, 32-bit row pointers, x.
                uint64_t csr_bytes = a.nnz() * 4 + (uint64_t)(m + 1) * 4 + (uint64_t)n * 2;
                uint64_t cyc_csr = nonPimCycles(csr_bytes, (uint64_t)m * 2);

                double secs =
                    chrono::duration<double>(chrono::steady_clock::now() - t0).count();
                csv << m << "," << n << "," << dist << "," << d << "," << a.nnz() << ","
                    << a.rowLengthCv() << "," << layout.rowParts << "x" << layout.colParts << ","
                    << layout.laneFill() << "," << layout.paddedPairs << "," << st.reads << ","
                    << st.xWrites << "," << st.crfWrites << "," << st.ctlWrites << ","
                    << st.flushWrites << "," << cyc_spmac << "," << cyc_gemv << ","
                    << cyc_host_dense << "," << cyc_csr << ","
                    << (double)cyc_gemv / cyc_spmac << "," << (double)cyc_csr / cyc_spmac << ","
                    << (double)cyc_host_dense / cyc_spmac << endl;
                cout << fixed << setprecision(3) << "  " << dist << " d=" << d
                     << " nnz=" << a.nnz() << " fill=" << layout.laneFill()
                     << " spmac=" << cyc_spmac << " csr=" << cyc_csr
                     << " vsGEMV=" << (double)cyc_gemv / cyc_spmac
                     << " vsCSR=" << (double)cyc_csr / cyc_spmac << " (" << secs << "s)"
                     << defaultfloat << endl;
            }
    }
}

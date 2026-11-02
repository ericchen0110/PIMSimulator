#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "MultiChannelMemorySystem.h"
#include "SparseExecution.h"
#include "SparseMapper.h"
#include "SparseMatrix.h"
#include "SystemConfiguration.h"

using namespace DRAMSim;

namespace
{
struct Options
{
    std::string matrix;
    std::string mapping = "all";
    std::string device = "ini/HBM2_samsung_2M_16B_x64.ini";
    std::string system = "system_hbm_1ch.ini";
    std::string csvPrefix;
    uint32_t channels = 1;
    uint32_t ranks = 1;
    uint32_t seed = 1;
    bool execute = true;
};

void usage(const char* program)
{
    std::cout << "Usage: " << program << " --matrix matrix.mtx [options]\n"
              << "  --mapping contiguous|round-robin|bank-aware|all\n"
              << "  --device PATH       device ini file\n"
              << "  --system PATH       system ini file\n"
              << "  --channels N        PIM channels used by the mapping\n"
              << "  --ranks N           PIM ranks per channel\n"
              << "  --seed N            deterministic input-vector seed\n"
              << "  --csv-prefix PATH   write summary, worker, traffic, and bank CSV files\n"
              << "  --mapping-only      skip functional PIM execution\n";
}

uint32_t parseUint(const std::string& value, const std::string& option)
{
    size_t consumed = 0;
    unsigned long parsed = std::stoul(value, &consumed);
    if (consumed != value.size() || parsed > UINT32_MAX)
        throw std::invalid_argument("invalid value for " + option + ": " + value);
    return static_cast<uint32_t>(parsed);
}

Options parseOptions(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index)
    {
        std::string argument = argv[index];
        auto value = [&](const std::string& option) -> std::string {
            if (index + 1 >= argc)
                throw std::invalid_argument("missing value for " + option);
            return argv[++index];
        };
        if (argument == "--matrix")
            options.matrix = value(argument);
        else if (argument == "--mapping")
            options.mapping = value(argument);
        else if (argument == "--device")
            options.device = value(argument);
        else if (argument == "--system")
            options.system = value(argument);
        else if (argument == "--channels")
            options.channels = parseUint(value(argument), argument);
        else if (argument == "--ranks")
            options.ranks = parseUint(value(argument), argument);
        else if (argument == "--seed")
            options.seed = parseUint(value(argument), argument);
        else if (argument == "--csv-prefix")
            options.csvPrefix = value(argument);
        else if (argument == "--mapping-only")
            options.execute = false;
        else if (argument == "--help" || argument == "-h")
        {
            usage(argv[0]);
            std::exit(0);
        }
        else
            throw std::invalid_argument("unknown option: " + argument);
    }
    if (options.matrix.empty())
        throw std::invalid_argument("--matrix is required");
    if (options.channels == 0 || options.ranks == 0)
        throw std::invalid_argument("--channels and --ranks must be positive");
    return options;
}

std::vector<SparseMappingPolicy> policies(const std::string& mapping)
{
    if (mapping == "all")
        return {SparseMappingPolicy::CONTIGUOUS, SparseMappingPolicy::ROUND_ROBIN,
                SparseMappingPolicy::BANK_AWARE};
    return {parseSparseMappingPolicy(mapping)};
}

std::vector<fp16> makeInput(uint32_t columns, uint32_t seed)
{
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<fp16> input(columns);
    for (fp16& value : input) value = convertF2H(distribution(generator));
    return input;
}

uint64_t phaseTransactions(const SparseRunStats& stats, SparsePhase phase)
{
    auto found = stats.traffic.find(phase);
    return found == stats.traffic.end() ? 0 : found->second.transactions;
}

void writeSummaryHeader(std::ofstream& output)
{
    output << "policy,rows,columns,nnz,workers,estimated_cost,ideal_cost,imbalance,cv,"
              "lane_padding,wave_padding,useful_segments,scheduled_segments,reload_rounds,"
              "setup_cycles,vector_cycles,compute_cycles,readback_cycles,spmv_cycles,total_cycles,"
              "setup_transactions,vector_transactions,compute_transactions,readback_transactions,"
              "execution_backend,predication,block_control,memory_scheduler,correct\n";
}

void writeWorkerHeader(std::ofstream& output)
{
    output << "policy,channel,rank,pim_block,rows,nnz,useful_segments,scheduled_segments,"
              "estimated_cost,effective_bank_bursts\n";
}

void writeTrafficHeader(std::ofstream& output)
{
    output << "policy,phase,request_class,transactions,reads,writes,bytes,"
              "effective_bank_bursts\n";
}

void writeBankHeader(std::ofstream& output)
{
    output << "policy,channel,rank,bank,controller_reads,controller_writes,activates\n";
}
}  // namespace

int main(int argc, char** argv)
{
    try
    {
        Options options = parseOptions(argc, argv);
        SparseMatrixCSR matrix = SparseMatrixCSR::loadMatrixMarket(options.matrix);
        std::vector<fp16> input = makeInput(matrix.numCols, options.seed);
        std::vector<fp16> reference = matrix.multiplyReference(input);

        std::ofstream summaryCsv;
        std::ofstream workerCsv;
        std::ofstream trafficCsv;
        std::ofstream bankCsv;
        if (!options.csvPrefix.empty())
        {
            summaryCsv.open((options.csvPrefix + ".summary.csv").c_str());
            workerCsv.open((options.csvPrefix + ".workers.csv").c_str());
            trafficCsv.open((options.csvPrefix + ".traffic.csv").c_str());
            bankCsv.open((options.csvPrefix + ".banks.csv").c_str());
            if (!summaryCsv || !workerCsv || !trafficCsv || !bankCsv)
                throw std::runtime_error("cannot open sparse benchmark CSV output");
            writeSummaryHeader(summaryCsv);
            writeWorkerHeader(workerCsv);
            writeTrafficHeader(trafficCsv);
            writeBankHeader(bankCsv);
        }

        for (SparseMappingPolicy policy : policies(options.mapping))
        {
            uint64_t memoryMb64 = static_cast<uint64_t>(256) * options.channels * options.ranks;
            if (memoryMb64 > UINT32_MAX || (memoryMb64 & (memoryMb64 - 1)) != 0)
                throw std::invalid_argument("configured total memory must be a power of two MB");
            auto memory = std::make_shared<MultiChannelMemorySystem>(
                options.device, options.system, ".", "sparse_bench",
                static_cast<uint32_t>(memoryMb64));
            if (std::getenv("PIMSIM_DEBUG") != nullptr)
            {
                SHOW_SIM_OUTPUT = true;
                DEBUG_CMD_TRACE = true;
            }
            uint32_t configuredChannels = getConfigParam(UINT, "NUM_CHANS");
            uint32_t pimBlocks = getConfigParam(UINT, "NUM_PIM_BLOCKS");
            if (configuredChannels != options.channels)
                throw std::invalid_argument("--channels must match NUM_CHANS in the system ini");

            SparseBankMapper mapper(options.channels, options.ranks, pimBlocks);
            SparseMappingPlan plan = mapper.map(matrix, policy);
            SparseMappingStats mappingStats = mapper.measure(matrix, plan);
            SparseExecutionResult execution;
            bool correct = true;
            if (options.execute)
            {
                HostGatherLockstepExecutor executor(memory, options.channels, options.ranks);
                execution = executor.run(matrix, plan, input);
                for (uint32_t row = 0; row < matrix.numRows; ++row)
                {
                    float actual = convertH2F(execution.values[row]);
                    float expected = convertH2F(reference[row]);
                    float tolerance = 0.02f * std::max(1.0f, std::fabs(expected));
                    if (std::fabs(actual - expected) > tolerance)
                    {
                        correct = false;
                        std::cerr << "row " << row << " mismatch: expected " << expected
                                  << ", got " << actual << '\n';
                    }
                }
            }

            std::cout << "\nSparse mapping: " << sparseMappingPolicyName(policy) << '\n'
                      << "  matrix: " << matrix.numRows << " x " << matrix.numCols << ", "
                      << matrix.nnz() << " nonzeros\n"
                      << "  workers: " << plan.workers.size() << '\n'
                      << "  estimated imbalance (max/mean): " << std::fixed
                      << std::setprecision(3) << mappingStats.maxToMeanImbalance << '\n'
                      << "  lane padding: " << mappingStats.lanePadding << '\n';
            if (options.execute)
            {
                std::cout << "  wave padding segments: "
                          << execution.stats.wavePaddingSegments << '\n'
                          << "  vector reload rounds: " << execution.stats.reloadRounds << '\n'
                          << "  recurring SpMV cycles: " << execution.stats.cycles.recurringSpmv
                          << '\n'
                          << "  end-to-end cycles: " << execution.stats.cycles.endToEnd << '\n'
                          << "  correctness: " << (correct ? "PASS" : "FAIL") << '\n';
            }

            if (summaryCsv)
            {
                const SparseRunStats& run = execution.stats;
                summaryCsv << sparseMappingPolicyName(policy) << ',' << matrix.numRows << ','
                           << matrix.numCols << ',' << matrix.nnz() << ',' << plan.workers.size()
                           << ',' << mappingStats.totalEstimatedCost << ','
                           << mappingStats.idealMaxEstimatedCost << ','
                           << mappingStats.maxToMeanImbalance << ','
                           << mappingStats.coefficientOfVariation << ','
                           << mappingStats.lanePadding << ',' << run.wavePaddingSegments << ','
                           << run.usefulSegments << ',' << run.scheduledSegments << ','
                           << run.reloadRounds << ',' << run.cycles.matrixSetup << ','
                           << run.cycles.vectorStaging << ',' << run.cycles.pimCompute << ','
                           << run.cycles.resultReadback << ',' << run.cycles.recurringSpmv << ','
                           << run.cycles.endToEnd << ','
                           << phaseTransactions(run, SparsePhase::MATRIX_SETUP) << ','
                           << phaseTransactions(run, SparsePhase::VECTOR_STAGING) << ','
                           << phaseTransactions(run, SparsePhase::PIM_COMPUTE) << ','
                           << phaseTransactions(run, SparsePhase::RESULT_READBACK)
                           << ",host-gather,off,lockstep,default,"
                           << (options.execute ? (correct ? 1 : 0) : -1) << '\n';
            }
            if (workerCsv)
            {
                for (size_t workerIndex = 0; workerIndex < mappingStats.workers.size(); ++workerIndex)
                {
                    const SparseWorkerStats& worker = mappingStats.workers[workerIndex];
                    uint64_t scheduled = workerIndex < execution.stats.workerScheduledSegments.size()
                                             ? execution.stats.workerScheduledSegments[workerIndex]
                                             : 0;
                    uint64_t effective =
                        workerIndex < execution.stats.workerEffectiveBankBursts.size()
                            ? execution.stats.workerEffectiveBankBursts[workerIndex]
                            : 0;
                    workerCsv << sparseMappingPolicyName(policy) << ',' << worker.worker.channel
                              << ',' << worker.worker.rank << ',' << worker.worker.pimBlock << ','
                              << worker.rows << ',' << worker.nnz << ',' << worker.segments << ','
                              << scheduled << ',' << worker.estimatedCost << ',' << effective
                              << '\n';
                }
            }
            if (trafficCsv)
            {
                for (const auto& phaseEntry : execution.stats.traffic)
                {
                    for (const auto& classEntry : phaseEntry.second.byClass)
                    {
                        const SparseTrafficCounter::ByClass& traffic = classEntry.second;
                        trafficCsv << sparseMappingPolicyName(policy) << ','
                                   << sparsePhaseName(phaseEntry.first) << ','
                                   << requestClassName(classEntry.first) << ','
                                   << traffic.transactions << ',' << traffic.reads << ','
                                   << traffic.writes << ',' << traffic.bytes << ','
                                   << traffic.effectiveBankBursts << '\n';
                    }
                }
            }
            if (bankCsv && options.execute)
            {
                uint32_t banks = getConfigParam(UINT, "NUM_BANKS");
                const MemoryStatsSnapshot& before = execution.stats.memoryBefore;
                const MemoryStatsSnapshot& after = execution.stats.memoryAfter;
                for (size_t channel = 0; channel < after.channels.size(); ++channel)
                {
                    const MemoryControllerSnapshot& start = before.channels[channel];
                    const MemoryControllerSnapshot& finish = after.channels[channel];
                    for (size_t flat = 0; flat < finish.readsPerBank.size(); ++flat)
                    {
                        bankCsv << sparseMappingPolicyName(policy) << ',' << channel << ','
                                << flat / banks << ',' << flat % banks << ','
                                << finish.readsPerBank[flat] - start.readsPerBank[flat] << ','
                                << finish.writesPerBank[flat] - start.writesPerBank[flat] << ','
                                << finish.activatesPerBank[flat] - start.activatesPerBank[flat]
                                << '\n';
                    }
                }
            }
            if (!correct)
                return 2;
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "sparse_bench: " << error.what() << '\n';
        usage(argv[0]);
        return 1;
    }
}

#ifndef SPARSE_PIM_ENERGY_H
#define SPARSE_PIM_ENERGY_H
#include <string>
#include "SparsePIM.h"
#include "MemoryController.h"

namespace DRAMSim
{
// An explicit event-based estimate, independent of the unused legacy counters.
// Dynamic coefficients: pJ/event. Static coefficients: mW/rank or mW/block.
struct SparsePIMEnergyModel
{
    std::string name;
    double readPJ = 0, writePJ = 0, activatePJ = 0, prechargePJ = 0, refreshPJ = 0;
    double macPJ = 0, instructionPJ = 0, rankMW = 0, blockMW = 0, independentMW = 0;
    static SparsePIMEnergyModel load(const std::string& filename);
    double dramPJ(const MemoryController::Activity&, double runtimeNS, unsigned ranks) const;
    double pimPJ(const SparsePIM::Stats&, double runtimeNS, unsigned blocks, bool independent) const;
};
}
#endif

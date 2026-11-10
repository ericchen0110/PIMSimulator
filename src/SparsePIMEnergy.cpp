#include "SparsePIMEnergy.h"
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <set>
#include <stdexcept>
using namespace DRAMSim;

SparsePIMEnergyModel SparsePIMEnergyModel::load(const std::string& filename)
{
    std::ifstream input(filename);
    if (!input) throw std::invalid_argument("Cannot open energy model: " + filename);
    SparsePIMEnergyModel m;
    std::map<std::string, double*> fields = {
        {"read_pj", &m.readPJ}, {"write_pj", &m.writePJ}, {"activate_pj", &m.activatePJ},
        {"precharge_pj", &m.prechargePJ}, {"refresh_pj", &m.refreshPJ}, {"mac_pj", &m.macPJ},
        {"instruction_pj", &m.instructionPJ}, {"rank_mw", &m.rankMW},
        {"block_mw", &m.blockMW}, {"independent_control_mw", &m.independentMW}};
    std::set<std::string> seen;
    std::string line;
    while (std::getline(input, line))
    {
        line = line.substr(0, line.find('#'));
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        std::istringstream stream(line);
        std::string key, value, extra;
        if (!(stream >> key >> value) || stream >> extra || !seen.insert(key).second)
            throw std::invalid_argument("Malformed/duplicate energy coefficient: " + line);
        if (key == "name") { m.name = value; continue; }
        if (!fields.count(key)) throw std::invalid_argument("Unknown energy coefficient: " + key);
        size_t used = 0;
        const double parsed = std::stod(value, &used);
        if (used != value.size() || !std::isfinite(parsed) || parsed < 0)
            throw std::invalid_argument("Invalid energy coefficient: " + key);
        *fields.at(key) = parsed;
    }
    if (seen.size() != fields.size() + 1 || m.name.empty() || m.name.find(',') != std::string::npos)
        throw std::invalid_argument("Energy model must name and specify every coefficient");
    return m;
}

double SparsePIMEnergyModel::dramPJ(const MemoryController::Activity& a, double ns, unsigned ranks) const
{
    return a.reads * readPJ + a.writes * writePJ + a.activates * activatePJ +
           a.precharges * prechargePJ + a.refreshes * refreshPJ + ns * ranks * rankMW;
}

double SparsePIMEnergyModel::pimPJ(const SparsePIM::Stats& s, double ns, unsigned blocks, bool independent) const
{
    return s.macs * macPJ + s.instructions * instructionPJ +
           ns * blocks * (blockMW + (independent ? independentMW : 0));
}

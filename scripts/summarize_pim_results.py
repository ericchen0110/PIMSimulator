#!/usr/bin/env python3
"""Compare paired PIM policies. Energy/power remain model estimates."""
import csv
import sys


def summarize(source, destination):
    pairs = {}
    with open(source, newline="") as stream:
        for row in csv.DictReader(stream):
            key = tuple(row[k] for k in ("group", "name", "pattern", "case_format",
                                         "execution_format", "system_config", "rows", "cols", "nnz",
                                         "seed", "mac_latency_cycles", "physical_blocks", "energy_model"))
            policy = row["control"]
            pair = pairs.setdefault(key, {})
            if policy not in ("lockstep", "independent") or policy in pair:
                raise ValueError("Invalid or duplicate policy for " + row["name"])
            pair[policy] = row
    with open(destination, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "name", "case_format", "system_config", "speedup",
                         "estimated_energy_reduction_pct", "estimated_power_reduction_pct", "energy_model"])
        for key, pair in pairs.items():
            if set(pair) != {"lockstep", "independent"}:
                raise ValueError("Missing matched policy for " + key[1])
            baseline, new = pair["lockstep"], pair["independent"]
            for field in ("macs", "reads", "writes"):
                if baseline[field] != new[field]:
                    raise ValueError("Work differs between policies: " + field)
            def ratio(field):
                old_value = float(baseline[field])
                return 1 - float(new[field]) / old_value if old_value else 0
            writer.writerow([key[0], key[1], key[3], key[5],
                             float(baseline["cycles"]) / float(new["cycles"]),
                             100 * ratio("estimated_total_energy_pj"),
                             100 * ratio("estimated_average_power_mw"), key[-1]])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: summarize_pim_results.py PAIRED_RESULTS.csv SUMMARY.csv")
    summarize(*sys.argv[1:])

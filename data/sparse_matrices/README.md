# Sparse SpMV benchmark inputs

`benchmark_cases.csv` is the canonical input set for the host-side sparse benchmark. Each row
describes a deterministic CSR matrix and the storage/memory configuration used for one run. Keeping
the seed fixed makes the generated matrices identical across baseline and future PIM experiments.

The suite covers density, size, sparsity pattern, row imbalance, vector locality, CSR/COO/ELL/BCSR
storage, and channel scaling. The road, social, and circuit cases are explicitly labeled
`application_proxy`: they reproduce characteristic sparsity behavior but are **not** third-party
SuiteSparse datasets. This avoids silently vendoring datasets with separate attribution or license
requirements.

Run every case from the repository root with:

```bash
scripts/run_sparse_benchmarks.sh
```

The default result is `results/sparse_benchmark_results.csv`. Pass another path as the first
argument to change it. PIMSimulator has no detailed host CPU or cache model, so `cycles` represents
HBM service time. Indirect vector accesses conservatively assume no host-cache hits.

## Sparse-aware scheduling

The benchmark labels transactions as sparse metadata, column index, matrix value, indirect input
vector, or output traffic. The opt-in memory-controller scheduler combines four signals:

- an open-row bonus to retain DRAM row-buffer locality;
- metadata/index priority and a dependency-progress bonus for matching index/value/vector groups
  when choosing among equally local requests;
- request age, including a starvation override after 1024 controller cycles; and
- FIFO order as the tie breaker.

Normal transactions are unchanged, and the controller uses the original FIFO admission policy
unless `MultiChannelMemorySystem::setSparseSchedulingEnabled(true)` is called. The sparse benchmark
does this by default. Run the same suite with the baseline policy using:

```bash
SPMV_SCHEDULER=baseline scripts/run_sparse_benchmarks.sh results/sparse_baseline.csv
```

Scheduler results include reordered transactions, row-hit admissions, average controller queue
wait, starvation promotions, and issued activate/precharge commands. These counters describe HBM
request scheduling; the benchmark still computes the numerical SpMV reference on the host.

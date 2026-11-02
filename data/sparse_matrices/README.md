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

# Independent sparse PIM control

The `independent_pim_control` branch adds an opt-in, autonomous sparse execution
model. It preserves the existing host-traffic benchmark and legacy CRF/SIMD
execution path. Sparse benchmark inputs and the original runner were copied from
`origin/sparse-aware-mem-scheduling`; that branch's simulator optimizations were
not merged.

## Run and compare

With SCons and GoogleTest installed:

```bash
# Original host HBM-traffic baseline, same defaults and CSV schema:
scripts/run_sparse_benchmarks.sh

# Same cases, two PIM control policies, separate CSV schema and output:
SPMV_BACKEND=pim scripts/run_sparse_benchmarks.sh results/pim_paired.csv
python3 scripts/summarize_pim_results.py results/pim_paired.csv results/pim_summary.csv

# Correctness, independent progress, completion, and energy-unit tests:
./sim --gtest_filter='SparsePIMControl.*'
```

Optional environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `SPMV_BACKEND` | `host` | `host` or `pim` |
| `SPMV_PIM_CONTROL` | `both` in runner | `both`, `lockstep`, `independent` |
| `SPMV_PIM_MAC_CYCLES` | `1` | MAC latency, equal for both policies |
| `SPMV_ENERGY_MODEL` | `data/sparse_matrices/pim_energy_example.txt` | Energy coefficient file |
| `SPMV_CASE_FILE` | Original `benchmark_cases.csv` | Alternative case list |
| `SPMV_MAX_CYCLES` | `100000000` | Watchdog per setup/execution/readback phase |
| `SPMV_SKIP_BUILD` | `0` | Reuse an existing **data-enabled** binary |
| `SPMV_BUILD_JOBS` | `8` | Build parallelism |
| `SPMV_GTEST_ROOT` | System paths | Optional GoogleTest prefix; forwarded as `GTEST_ROOT` |

`NO_STORAGE=1` is deliberately rejected by the sparse engine: it cannot validate
or traverse sparse metadata without stored data. A standalone configured test
uses `SPMV_PIM_CONTROL=lockstep` by default; the script explicitly runs both.
The sparse datapath always uses FP32; `PIM_PRECISION` continues to control the
legacy SIMD datapath only.

In this workspace, validation dependencies were installed under `/tmp` rather
than changing the system installation. Rebuild with:

```bash
PYTHONPATH=/tmp/pim-build-deps /tmp/pim-build-deps/bin/scons -j4 GTEST_ROOT=/tmp/pim-test-deps
```

Use `SPMV_SKIP_BUILD=1` to run the already-built binary without SCons in `PATH`.

## Hardware/software boundary

The fixture generates the same seeded matrix as the host baseline, assigns rows
statically round-robin to block bank pairs, and loads actual FP32 values, CSR
indices, row pointers, and a replicated input vector. It submits one
`SparsePIMJob`, then only calls `memory.update()` until completion. It reads
actual simulated output and compares every row against a host reference.
There is no host-side per-block issue schedule, zero skipping, or runtime pause
logic. Setup and validation traffic are excluded from the timed execution region.

`SparsePIM` is owned and clocked by `MultiChannelMemorySystem`. Each participating
block has a PC, row and nonzero cursors, accumulator, operand buffer, outstanding
request flag, and MAC readiness cycle. Its fixed sparse microprogram is:

```
read row bounds -> read column -> read value -> read x[column] -> MAC
                                      ^                         |
                                      +------ next nonzero -----+
                   -> store row result -> next row -> done
```

Only one request is outstanding per block. Every request goes through the existing
transaction queues, command scheduler, bank timing, refresh, and data return path.
Queue backpressure leaves the issuing block's PC unchanged. Completion hooks
wake the correct block after a read returns or a write commits. Pending-work
checks include autonomous execution and queued transactions.

Lockstep policy synchronizes nonzero slots and row boundaries **within each
channel/rank**, masking exhausted rows until peers catch up. Independent policy
removes these barriers. Both policies use exactly the same mapping, memory
requests, FP32 arithmetic order, and MAC latency. Empty rows produce zero; blocks
with no assigned rows finish immediately. There is no work stealing, cross-block
reduction, automatic dependency discovery, or speculative execution. SpMV rows
are independent by construction; dependent kernels require another descriptor
and explicit synchronization semantics.

This is a new scalar FP32 sparse datapath/control model, not an implementation of
independently executing arbitrary legacy CRF instructions. Its PC is a sparse
microprogram state, not the legacy rank's CRF index. The original `PIMRank` shared
PC and SIMD behavior are untouched. Operand requests conservatively use the
existing memory-controller/data-bus path; this does not model a separate internal
bank-to-PIM interconnect or claim native PIM bandwidth. Absolute hardware timing
requires calibration and a more detailed datapath/interconnect model.

## Fair comparisons and integration

Use lockstep PIM as the denominator for the **control-only** speedup. The host
baseline measures HBM service traffic with no CPU computation timing; it is not
an end-to-end CPU SpMV baseline. The PIM path keeps one rank per channel as channel
count scales; the original host baseline keeps its original fixed total capacity.
Compare policies within the PIM path, not their raw cycles against host CPU time.

Every case currently executes CSR. `case_format` preserves the original case's
requested CSR/COO/ELL/BCSR label; `execution_format=csr` states the actual device
layout. Format cases reuse the same generated matrix but do not measure native
COO/ELL/BCSR execution or conversion costs. No format-specific speedup should be
inferred from these rows.

For combining branches:

- Mapping code can supply another static row assignment and the job's `address`
  adapter. Loading and runtime address generation must use the same adapter.
- Gather can replace the indexed-vector operand fetch behind the sparse engine's
  `Input` stage, keeping readiness/completion semantics.
- Memory scheduling continues to arbitrate the engine's ordinary requests. Keep
  transaction completion hooks and monotonic activity counters when merging.
- Compare each feature configuration with identical jobs, input placement, and
  energy coefficients before enabling combinations. A faster control policy does
  not guarantee lower energy or average power.

## Energy and power

The original `burstEnergy`/`actpreEnergy` counters are not populated in this
checkout. The new benchmark therefore uses separate monotonic **issued command**
counts and an explicit estimate:

```
DRAM pJ = sum(command count * command pJ) + runtime ns * ranks * rank mW
PIM pJ  = MACs * MAC pJ + instructions * instruction pJ
          + runtime ns * blocks * (block mW + independent-control mW when enabled)
average mW = total pJ / runtime ns
```

The model includes all physical block/rank static costs, including idle blocks,
and does not assume perfect clock/power gating. Instruction events are accepted
operand/output requests and completed MACs. Control flow costs beyond that are
represented only by static power; caches, DQ versus internal transfer distinctions,
and detailed control-area overhead require a calibrated model.

The bundled coefficients are **illustrative and uncalibrated**, not device
measurements. Replace every coefficient with supported values before reporting
hardware power savings. The CSV names all energy/power columns `estimated_*` and
records the model name, seed, MAC latency, physical block count, memory-wait and
barrier block-cycles, and maximum row progress difference. Give changed coefficient
sets distinct model names. Negative reductions in the summary mean regressions.

## Validation and saved results

`results/independent_pim_paired.csv` contains both policies for all 30 canonical
cases; `results/independent_pim_summary.csv` contains paired comparisons.
`results/independent_pim_host_baseline.csv` preserves the host baseline run.
The results use the bundled illustrative model and default one-cycle MAC latency.
Across the 30 cases, independent execution is faster in 29, with a geometric-mean
speedup of 1.1451x and a range of 0.9820x–1.4121x. Estimated energy is lower in 7
cases; energy reductions range from -7.192% to +0.882%. These figures describe
this model and its chosen coefficients, not measured hardware savings.
The targeted uneven-row test uses a three-cycle MAC and confirms that short-row
blocks execute multiple rows ahead, with equal results and read/write counts.
Additional tests cover empty rows, an all-zero matrix, partial final block waves,
signed arithmetic, duplicate-address read completions, job submission guards,
and energy unit conversion.

Legacy kernel correctness and performance tests were also run. The pre-existing
`PIMBenchFixture.add` threshold test reports 1.985965967x versus its required >2x;
a pristine `dev` build reproduces exactly the same failure and cycle count.

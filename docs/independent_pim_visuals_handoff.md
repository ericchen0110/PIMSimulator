# Handoff to ChatGPT: independent PIM-block control visuals

Use this document as a self-contained brief. It contains the implementation
explanation, available numerical evidence, visual specifications, and a prompt
to start designing the figures. No access to the original coding conversation is
required.

## Copy-ready starting prompt

> Help me create a small set of clear, technically accurate visuals for an
> EECS computer architecture project about independent PIM-block control.
> Read this entire brief before designing anything. Create:
>
> 1. An architecture diagram showing what the benchmark supplies and what the
>    simulator's hardware model controls.
> 2. A side-by-side illustrative execution timeline for lockstep and independent
>    blocks, showing how short rows can advance while another block is busy.
> 3. A performance figure using the supplied benchmark data and aggregate results.
> 4. A figure separating execution time, estimated energy, and average power.
> 5. A compact explanation of the sparse block's PC, memory stalls, and completion.
>
> Start with a proposed layout, titles, captions, and speaker notes. Then generate
> the diagrams and charts. Use editable diagrams and reproducible plotting code
> where possible; export charts as SVG and high-resolution PNG. Use the embedded
> numbers directly. Do not invent missing per-case results, execution traces,
> confidence intervals, or hardware power measurements. Visually distinguish
> conceptual illustrations from recorded simulation results. Keep the actual
> comparison baseline and model limitations clear. Highlight useful improvements
> while retaining the observed regressions.

## Provenance and current availability

This brief was assembled on **2026-11-08** from the recorded implementation and
validation session earlier in this conversation. The recorded branch name was
`independent_pim_control`.

At handoff creation, the accessible checkout is back on `dev`; the implementation
branch and its saved CSVs are absent. The legacy shared-PC code and canonical
30-case benchmark list on `origin/sparse-aware-mem-scheduling` can still be
inspected. Implementation details and experimental numbers below are **records
from the earlier session**, not a newly reproduced run. Do not describe the
implementation as currently verified in this checkout.

The numerical tables embedded here are sufficient for the proposed selected-case
and targeted-test visuals. A complete 30-case bar chart requires recovering the
full CSVs. Do not fill in their missing rows from the aggregate statistics.

Recorded artifact names, for recovery if available elsewhere:

- `results/independent_pim_paired.csv`: 60 rows, two policies for 30 cases.
- `results/independent_pim_summary.csv`: 30 paired comparison rows.
- `results/independent_pim_host_baseline.csv`: 30 host-memory baseline rows.
- `docs/independent_pim_control.md`: implementation and model documentation.

## The problem and the implemented approach

Processing-in-memory (PIM) places computation near memory banks. In the original
simulator, `PIMRank` stores one CRF program counter per channel/rank. A qualifying
host memory packet triggers an instruction, which is executed across every PIM
block in that rank before the PC advances. The host kernel also supplies operand
addresses and vector tiles. Merely adding per-block PCs would not supply the
right addresses, preserve operands, or model readiness independently.

Sparse matrix-vector multiplication computes:

```
y[row] = sum(value[k] * x[column[k]]) for each nonzero k in row
```

In CSR, row pointers identify each row's nonzeros; column indices identify vector
operands. Implicit zeros are already absent. **This experiment primarily targets
unequal row lengths and memory stalls.** It does not claim to discover zeros in
a dense matrix or eliminate all memory accesses associated with zeros.

The implementation added a **new scalar FP32 CSR controller** inside the
simulator. Each block has its own sparse microprogram PC and data-traversal state.
The simulator chooses when it issues, waits, computes, and advances. The benchmark
loads data and submits a job; it does not schedule individual blocks at runtime.

Two policies run the same sparse controller:

| Policy | Behavior |
|---|---|
| Lockstep PIM | Rank-local barriers synchronize sparse nonzero slots and row boundaries. Exhausted rows are masked while peers finish. |
| Independent PIM | Each block advances when its own operands and execution resources are ready, without those barriers. |

This matched lockstep policy is the denominator for the reported control-only
speedups. **It is distinct from the untouched legacy CRF interpreter.** The
experiment does not independently execute arbitrary CRF programs and does not
compare the new scalar controller directly against legacy SIMD throughput.

## Architecture details to draw

### Benchmark/software responsibilities

1. Generate a deterministic matrix and vector.
2. Assign rows statically, round-robin, to PIM block bank pairs.
3. Load actual FP32 values, CSR row pointers/indices, and a replicated input vector.
4. Submit a `SparsePIMJob` and select lockstep or independent control.
5. Repeatedly advance the simulator clock until the job completes.
6. Read actual simulated output and compare every row with a host reference.

Static placement is software work. Runtime issue, skipping exhausted work,
stalling, synchronization, and completion are hardware-model work.

### Simulator/hardware-model responsibilities

`MultiChannelMemorySystem` owns and clocks `SparsePIM`. Each participating block
has:

- PC / sparse microprogram state;
- local row number, nonzero cursor, and row-end cursor;
- current column index, value, and vector operand;
- FP32 accumulator;
- one outstanding memory-request flag and a burst buffer;
- a MAC readiness cycle and completion state.

The fixed microprogram follows:

```
Read row start/end pointers
  -> row empty? -------------------------------> Store zero result
  -> Read column index
  -> Read matrix value
  -> Read x[column]
  -> Wait for MAC readiness, then accumulate
  -> More nonzeros? -> repeat from Read column index
  -> Store result
  -> Next row? -> repeat from Read row pointers
  -> Done
```

Lockstep policy adds sparse-slot and row-boundary synchronization. Independent
policy removes those waits. Keep these synchronization waits visually separate
from waits for memory data.

Memory requests use the existing transaction queues, command scheduler, DRAM
bank timing, refresh, and response path. Queue backpressure leaves the block's PC
unchanged. Read completion makes operands ready; write completion follows the
actual simulated write. Job completion includes all participating blocks and
outstanding output writes.

The recorded implementation used **one outstanding request per block**, one
scalar FP32 nonzero per MAC issue, and one output burst per row. Do not draw
unlimited concurrent operations within one block.

Suggested architecture diagram:

```
[Benchmark: generate / place / load / submit]
                        |
                  [SparsePIMJob]
                        |
       +--- simulator / modeled hardware boundary ---+
       | [Clocked sparse controller + policy]        |
       |    |             |             |           |
       | [Block 0]     [Block 1]     [Block ...]      |
       | PC/cursors    PC/cursors    PC/cursors      |
       | accumulator   accumulator   accumulator    |
       |    | requests    | requests    | requests  |
       | [Existing memory queues / scheduler / DRAM] |
       |    +--------- completion responses ---------|
       +--------------------------------------------+
                        |
              [Read back and validate]
```

Use an inset for the legacy shared-CRF-PC design if useful. Label the inset
“legacy reference architecture,” so viewers do not mistake it for the measured
lockstep sparse baseline.

### Combining teammates' optimizations

Show three possible integration points as dashed extension boxes, labeled
“future composition; combined results not measured”:

- **Bank-aware mapping:** static placement and the job's address adapter.
- **Indexed gather:** the `Read x[column]` operand-fetch stage.
- **Memory scheduling:** arbitration of the controller's ordinary memory requests.

No combined speedup or power reduction was measured in the recorded session.
Do not multiply independent speedups or draw a measured “all optimizations” bar.

## Recorded results: canonical 30-case suite

All 30 cases ran under both policies. Each run validated its actual simulated
outputs. The PIM execution layout was CSR for every case, including cases whose
original labels requested COO/ELL/BCSR. Thus format-labeled cases do **not** show
native COO/ELL/BCSR execution benefits.

The suite varies density, size, pattern, row imbalance, vector locality,
application proxies, requested format, and channel count. The application cases
are synthetic road/social/circuit proxies, not external application datasets.

| Aggregate | Recorded value |
|---|---:|
| Matched cases | 30 |
| Independent faster than lockstep | 29 / 30 |
| Geometric-mean speedup | 1.1451287679x |
| Minimum speedup | 0.9819871220x |
| Maximum speedup | 1.4121401366x |
| Cases with lower estimated total energy | 7 / 30 |
| Minimum estimated energy reduction | -7.1921855018% |
| Maximum estimated energy reduction | +0.8821382923% |

Use **1.145x geometric-mean speedup** as the rounded headline. Its corresponding
normalized execution-time reduction is approximately 12.7%; do not label 1.145x
as a 14.5% reduction in runtime. Preserve the slight slowdown and energy
regressions in the supporting figure.

These are deterministic simulator observations. No repeated-run variability or
confidence intervals were recorded.

### Selected case comparisons available in the conversation

This is a **selected five-case subset**, not the complete suite and not an
independent sample. It includes a negligible gain and a regression. The aggregate
above must not be recomputed from these five rows.

```csv
group,name,channels,speedup,estimated_energy_reduction_pct,estimated_power_reduction_pct
imbalance,uniform_rows,16,1.1081714206972577,-1.0627906100588058,-11.994896249978604
imbalance,moderate_skew,16,1.412140136567958,-2.1423105022422506,-44.23925630192509
imbalance,heavy_skew,16,1.0046437286641716,-0.6112506200476542,-1.0784619684589547
scaling,channels_1,1,1.196311870373182,-2.963848312982198,-23.176873955995614
scaling,channels_64,64,0.9819871220148342,-0.9912493501396336,0.8278937021132804
```

The three imbalance cases use 1024x1024 matrices; `uniform_rows` and
`moderate_skew` use the case parameter `nnz_per_row=16`, while `heavy_skew` uses
8. The two channel cases use 2048x2048 uniform matrices and `nnz_per_row=16`.
All five use seed 570, CSR, and one-cycle MAC latency. Pattern-specific generators
can make actual row lengths differ from the `nnz_per_row` parameter.

Do not infer the cause of a particular case's result from these numbers alone.
For example, the heavy-skew label does not guarantee the largest gain: static
ownership, the longest block's work, and memory contention also matter.

## Recorded results: targeted uneven-row correctness test

This is a **separate mechanism-demonstration test**, outside the canonical
30-case suite. It deliberately rotates long rows between blocks across row waves.
It uses 32 rows, 64 columns, 284 nonzeros, one channel, one rank, eight blocks,
and a **three-cycle MAC**, rather than the suite's one-cycle MAC.

```csv
policy,cycles,runtime_ns,reads,writes,macs,memory_wait_block_cycles,barrier_block_cycles,max_row_lead,estimated_dram_energy_pj,estimated_pim_energy_pj,estimated_total_energy_pj,estimated_average_power_mw
lockstep,35731,35731,916,32,284,40692,242928,1,402931,4117.68,407048.68,11.3920315692
independent,9417,9417,916,32,284,39142,0,4,369137,2087.896,371224.896,39.4207174259
```

Useful labels derived from this table:

- **3.7943x speedup**, or **73.64% fewer execution cycles**.
- Equal outputs and equal counts: 916 reads, 32 writes, 284 MACs.
- **8.80% lower estimated total energy** with the illustrative coefficients.
- **246.04% higher estimated average power**: lower energy does not imply lower
  average power when runtime shrinks much faster.
- Maximum local-row progress difference: 1 in lockstep, 4 in independent mode.
- Synchronization-wait block-cycles: 242,928 versus 0. These are summed across
  blocks and are not elapsed cycles or an execution trace.

Use the 3.79x result as a labeled targeted example, not as the suite-wide headline.
Do not claim it is a trace from a real device.

## Energy model and comparison rules

The original energy counters in this checkout were unpopulated, and device
current parameters were zero. The implementation therefore used monotonic issued
memory-command counts and an explicit configurable energy estimate.

The coefficient set was called `illustrative_uncalibrated_v1`:

| Coefficient | Value | Unit |
|---|---:|---|
| Read | 100 | pJ/event |
| Write | 120 | pJ/event |
| Activate | 200 | pJ/event |
| Precharge | 80 | pJ/event |
| Refresh | 1000 | pJ/event |
| Scalar MAC | 4 | pJ/event |
| Instruction event | 0.1 | pJ/event |
| Rank static power | 1 | mW/rank |
| Block static power | 0.01 | mW/block |
| Additional independent-control static power | 0.001 | mW/block |

An instruction event means an accepted operand/output request or completed MAC.
All physical ranks/blocks incur their configured static cost, including idle
blocks. The model does not assume perfect power gating.

```
DRAM pJ = sum(issued command count * command pJ)
          + runtime ns * physical ranks * rank mW

PIM pJ = MAC count * MAC pJ + instruction events * instruction pJ
         + runtime ns * physical blocks
           * (block mW + extra control mW in independent mode)

estimated total pJ = DRAM pJ + PIM pJ
estimated average mW = estimated total pJ / runtime ns

speedup = lockstep cycles / independent cycles
energy reduction % = 100 * (1 - independent energy / lockstep energy)
power reduction % = 100 * (1 - independent power / lockstep power)
```

A negative reduction means an increase. For normalized plots, set lockstep to 1:

```
independent normalized time   = 1 / speedup
independent normalized energy = 1 - energy_reduction_pct / 100
independent normalized power  = 1 - power_reduction_pct / 100
```

Always label energy/power as **estimated, uncalibrated**. These coefficients are
illustrative, not Samsung/HBM measurements. Changing calibrated coefficients may
change the energy conclusions. Do not claim validated hardware power savings.

The timed region excludes input loading and output readback/validation. It
includes device traversal, operand memory accesses, computation, synchronization,
and output writes. Each policy receives identical matrix/vector data, placement,
MAC latency, and energy coefficients. PIM runs use one rank per channel.

The existing host-memory benchmark measures HBM service traffic; it has no host
CPU computation-time model. Its fixed total capacity also differs from the
PIM experiment's one-rank-per-channel scaling. Use matched lockstep PIM for
control-only speedup, not the host-traffic baseline as a CPU SpMV denominator.

## Visual specifications

### 1. Architecture: “Runtime control moves into the simulator”

Draw the software/model boundary and the request/completion arrows described
above. Show a PC and readiness state in each block. For the policy difference,
use a rank-local synchronization line in lockstep and independent readiness
arrows in independent mode. Include the future integration points as dashed
extensions. Caption: “Software supplies data and a job; modeled hardware chooses
when each sparse block advances.”

### 2. Timeline: “A short row can advance while a peer finishes”

Use a schematic with four block lanes and two rows per block. The following
**invented teaching example** supplies MAC-work lengths only:

```csv
block,row0_work_slots,row1_work_slots
0,8,1
1,1,8
2,2,2
3,0,3
```

Ignoring all memory/metadata/store costs, lockstep row waves require 8 + 8 = 16
work slots, while independent lanes finish by max(9,9,4,3) = 9 slots. Render the
short-row waits in gray and mark the 8-slot row boundary only in lockstep.
A zero-length row still requires a real result store in the actual implementation;
that cost is deliberately omitted from this schematic.

Label prominently: **“Illustrative work slots; not simulator cycles or a measured
trace.”** Keep these illustrative numbers separate from the recorded targeted
test. Do not fabricate a cycle-accurate Gantt chart from aggregate wait counters.

### 3. Performance: “Independent control improves most tested cases”

Use a horizontal speedup bar chart for the five supplied cases, with a vertical
reference at 1.0x and a visible axis region below 1.0x. Keep the 64-channel slowdown
visible. Add separate suite-wide callouts: 29/30 faster; 1.145x geometric mean;
0.982x–1.412x range. Label the bars “selected cases.”

If the full CSV is recovered, replace the subset with all 30 cases grouped by
benchmark category. Do not append the targeted 3.79x result to the canonical
aggregate or portray the min/max range as a confidence interval.

### 4. Tradeoff: “Faster execution can increase average power”

Use the targeted test for three aligned panels or a grouped normalized chart:
execution time, estimated total energy, and estimated average power. Set lockstep
to 1 in each metric; independent is approximately 0.264, 0.912, and 3.460.
Use the raw values in the table to compute the exact bars. Annotate 3.79x speedup,
8.80% less estimated energy, and 246% more estimated average power. Caption the
three-cycle MAC and targeted-test status.

Optional second panel: a selected-case scatter plot of speedup versus estimated
energy reduction, with reference lines at 1x and 0%. This makes regressions
visible without suggesting all faster cases save energy.

### 5. Control flow: “Each block waits only for its own dependencies”

Show the sparse microprogram as an editable state diagram. Add a memory-wait
self-loop and a queue-backpressure retry. Depict the lockstep barriers as policy
hooks that are bypassed in independent mode. Mark output-write completion before
“row done” and all blocks done before “job done.” Keep this diagram distinct from
an arbitrary-CRF instruction decoder.

## Presentation and accuracy checklist

- Audience: computer architecture classmates and instructors familiar with
  banks, PCs, and matrix multiplication; explain CSR briefly.
- Suggested palette: blue lockstep, orange independent, gray synchronization wait,
  a distinct hatch/outline for memory wait. Use labels as well as colors.
- Use readable fonts and short titles. Offer one-sentence captions and 2–3
  speaker-note sentences per figure.
- Export editable SVG or diagram source plus PNG; use reproducible plotting tools
  for numerical charts and avoid generated images for chart text and exact data.
- Do not invent error bars, trace events, absolute cycle counts for missing cases,
  physical area, silicon measurements, or combined-optimization results.
- The measured model is a scalar FP32 CSR extension. The legacy SIMD CRF path was
  preserved; arbitrary legacy programs do not acquire independent PCs.
- Operand traffic uses the existing memory-controller/data-bus timing path. A
  dedicated internal bank-to-PIM interconnect was not modeled. Treat absolute
  hardware timing and power claims as outside the demonstrated scope.
- CSR omits implicit zeros. Distinguish independent progress from a separate
  dense zero-detection or arithmetic-gating optimization.
- Static partitioning remains. There is no work stealing, speculative execution,
  cross-block reduction, or general dependency discovery. SpMV rows are
  independent by construction.
- Recorded validation: 11 final correctness checks passed, and all 60 canonical
  PIM runs validated outputs. An existing ADD speedup-threshold test failed at
  1.985965967x against a required >2x; pristine `dev` reproduced the same result.
- Prefer this claim: “The simulator-owned sparse controller lets blocks advance
  independently and improves performance on most evaluated cases; energy and
  power depend on workload and the selected estimation model.”

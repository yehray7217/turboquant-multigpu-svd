# TQ Kernel Microbenchmark

This experiment isolates the `mode=tq` device payload path from the full RSVD
pipeline. It does not run QJL.

The benchmark measures:

```text
encode      = norm + Rademacher sign + FWHT + Lloyd-Max quantize + bitpack
decode_add  = Lloyd-Max decode + inverse FWHT + sign/norm add
roundtrip   = encode + decode_add
```

It also has a profiling-only path that uses CUDA events to split TQ into:

```text
clear_codes
norm
transform
quantize_bitpack
decode_event
encode_event_total
roundtrip_event_total
```

The CUDA event totals are the numbers to use for phase attribution. The
host-side `encode`/`decode_add` lines are kept for rough continuity with older
runs, but they include profiling helper overhead in this breakdown mode.

It also reports:

```text
relative_error = ||A - decode(encode(A))||_2 / ||A||_2
```

## Why This Exists

Full RSVD timing includes GEMM, SVD, peer copies, host staging, MPI, and
synchronization. That makes it hard to tell whether a CUDA change helped the TQ
kernel itself.

This microbenchmark is the next profiling step after the successful fused-FWHT
optimization. It should help decide whether the next TQ-only optimization should
target:

```text
column norm
fused FWHT
Lloyd-Max bucket lookup
bitpack write
decode/add
```

## Run

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_tq_kernel_microbench
module load cuda/12.8
make
sbatch run_tq_kernel_microbench.slurm
```

Then inspect:

```bash
ls -lt tq_kernel_microbench_*.out
cat tq_kernel_microbench_<jobid>.out
```

## Default Cases

The slurm script runs:

| rows | cols | bits | warmup | repeat | Purpose |
|---:|---:|---:|---:|---:|---|
| 256 | 16384 | 4 | 20 | 200 | Current RSVD-scale TQ vector count |
| 256 | 32768 | 4 | 20 | 200 | Larger vector count for scaling/noise reduction |

## Interpretation

Use the `encode` and `decode_add` lines to decide where the next CUDA attempt
should focus.

If `encode` dominates, likely candidates are:

```text
column_norms_kernel
column_tq_normalize_sign_fwht256_kernel
column_tq_lloyd_quantize_kernel
bitpack_write_code
```

If `decode_add` dominates, likely candidates are:

```text
column_tq_lloyd_dequantize_fwht256_apply_add_kernel
bitpack_read_code
```

QJL is intentionally excluded from this experiment.

## Result: 2026-06-02

Run:

```text
job_id = 942708
node = gn1220
GPU = 1 x V100
bits = 4
warmup = 20
repeat = 200
```

| rows | cols | FP32 bytes | code bytes | norm bytes | encode | decode_add | roundtrip | relative error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 16384 | 16.000 MiB | 2.000 MiB | 0.062 MiB | 0.2325 ms | 0.0875 ms | 0.3200 ms | 0.09717795 |
| 256 | 32768 | 32.000 MiB | 4.000 MiB | 0.125 MiB | 0.4089 ms | 0.1543 ms | 0.5631 ms | 0.09720906 |

Encode dominates the TQ roundtrip:

| rows | cols | encode share | decode share |
|---:|---:|---:|---:|
| 256 | 16384 | 72.7% | 27.3% |
| 256 | 32768 | 72.6% | 27.4% |

The next optimization should focus on encode, not decode. The remaining encode
pipeline is:

```text
column_norms_kernel
column_tq_normalize_sign_fwht256_kernel
column_tq_lloyd_quantize_kernel
```

The full RSVD timing is much larger than this microbenchmark because it also
includes GEMM/SVD, payload movement, peer copies, host staging, MPI, and global
synchronization. This benchmark confirms that the TQ codec itself is sub-ms at
the current `l=256, n=16384` scale, so further TQ-only CUDA optimizations will
need to be targeted carefully.

## Result: Encode Phase Breakdown

Run:

```text
node = gn1220
GPU = 1 x V100
bits = 4
warmup = 20
repeat = 200
```

| rows | cols | clear_codes | norm | transform | quantize_bitpack | decode_event | encode_event_total | roundtrip_event_total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 16384 | 0.0072 ms | 0.0562 ms | 0.0746 ms | 0.0987 ms | 0.0844 ms | 0.2368 ms | 0.3212 ms |
| 256 | 32768 | 0.0095 ms | 0.0949 ms | 0.1322 ms | 0.1694 ms | 0.1490 ms | 0.4059 ms | 0.5549 ms |

Encode phase share:

| rows | cols | clear_codes | norm | transform | quantize_bitpack |
|---:|---:|---:|---:|---:|---:|
| 256 | 16384 | 3.0% | 23.7% | 31.5% | 41.7% |
| 256 | 32768 | 2.3% | 23.4% | 32.6% | 41.7% |

The largest encode phase is `quantize_bitpack`, followed by the fused transform.
The next CUDA experiment should target quantization/bitpacking first. However,
the previous full-pipeline specialized TQ4 pack attempt was slower because it
reduced parallelism. Any new pack experiment should first be isolated in this
microbenchmark before touching the main RSVD path.

## Pending: Alternative TQ4 Quantize/Bitpack Microbenchmark

The next microbenchmark-only experiment should compare the current generic
one-thread-per-coordinate atomic bitpack against an alternative TQ4 byte-pack
kernel. This is not yet a main-pipeline optimization.

Expected output should include:

```text
quantize_bitpack
quantize_pack4_alt
```

Only if the alternative wins in the microbenchmark should it be considered for
the main TQ path.

The alternative is intentionally measured after the normal roundtrip so it does
not affect the reported `relative_error`. It reuses the transformed `d_work`
buffer from encode and overwrites `d_codes` only for timing.

## Result: Alternative TQ4 Quantize/Bitpack

Run:

```text
GPU = 1 x V100
bits = 4
warmup = 20
repeat = 200
```

| rows | cols | quantize_bitpack | quantize_pack4_alt | Result |
|---:|---:|---:|---:|---|
| 256 | 16384 | 0.0984 ms | 0.0992 ms | alt is slightly slower |
| 256 | 32768 | 0.1673 ms | 0.1668 ms | alt is only 0.0005 ms faster |

The alternative pack4 kernel does not provide a meaningful win. It removes
`atomicOr`, but it also makes each thread process two coordinates and perform
two Lloyd-Max bucket searches. The net result is essentially tied with the
current generic one-thread-per-coordinate bitpack path.

Decision: **do not move pack4_alt into the main TQ path**. The next TQ-only
target should not be bitpack unless a different design preserves
one-thread-per-coordinate parallelism while reducing atomic pressure.

## Nsight Compute Profiling

The next step is to profile the real TQ kernels with Nsight Compute instead of
guessing from timings alone. Use:

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_tq_kernel_microbench
sbatch run_ncu_tq_kernels.slurm
```

The script profiles one launch each for:

| Label | Kernel |
|---|---|
| `norm` | `column_norms_kernel` |
| `fused_fwht256` | `column_tq_normalize_sign_fwht256_kernel` |
| `quantize_bitpack` | `column_tq_lloyd_quantize_kernel` |
| `decode_add` | `column_tq_lloyd_dequantize_fwht256_apply_add_kernel` |

The script writes:

```text
tq_kernel_ncu_<jobid>.out
tq_kernel_ncu_<jobid>.err
ncu_reports/*.ncu-rep
```

The stdout should contain a text summary. The `.ncu-rep` files can be opened in
Nsight Compute UI if needed.

If stdout only shows the application output and report paths, export a compact
text summary from the generated reports:

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_tq_kernel_microbench
module load cuda/12.8
bash summarize_ncu_reports.sh ncu_reports/*_<jobid>.ncu-rep > ncu_summary_<jobid>.txt
cat ncu_summary_<jobid>.txt
```

Do not interpret the application timing printed while running under `ncu` as
normal runtime. Nsight Compute replays kernels across many profiling passes, so
the reported `encode`/`decode_add` times can inflate from sub-ms to hundreds of
ms. Use the `.ncu-rep` metrics for bottleneck analysis.

Use these fields to decide the next optimization:

| Signal | Meaning | Likely next step |
|---|---|---|
| Low SM utilization, high memory throughput | Memory bound | reduce global reads/writes or improve coalescing |
| Low occupancy | Too few active warps | adjust block shape/register/shared-memory usage |
| High atomic or serialization symptoms in quantize | bitpack pressure | redesign bitpack without reducing coordinate parallelism |
| High branch/warp stalls | divergent bucket search | revisit Lloyd-Max lookup design |

Do not use QJL runs for this step.

## Result: Nsight Compute Report Capture

Run:

```text
job_id = 942924
node = gn1216
GPU = 1 x V100
ncu = 2025.1.0.0
```

Reports generated:

```text
ncu_reports/norm_942924.ncu-rep
ncu_reports/fused_fwht256_942924.ncu-rep
ncu_reports/quantize_bitpack_942924.ncu-rep
ncu_reports/decode_add_942924.ncu-rep
```

The run successfully captured reports, but the stdout did not include enough
metric detail for analysis. Export the report metrics with
`summarize_ncu_reports.sh` before choosing the next CUDA optimization.

## Result: Nsight Compute Summary

Run:

```text
job_id = 942924
node = gn1216
GPU = 1 x V100
ncu = 2025.1.0.0
```

Compact report:

| Kernel | Memory Throughput | Mem Busy | Max Bandwidth | L1/TEX Hit | L2 Hit | Active Threads / Warp | Not Predicated / Warp | Achieved Occupancy | Key Signal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `decode_add` | 413.91 GB/s | 64.35% | 52.35% | 63.45% | 50.67% | 32.00 | 30.78 | 92.45% | shared-memory scoreboard stalls |
| `fused_fwht256` | 439.90 GB/s | 68.13% | 54.62% | 9.72% | 50.70% | 32.00 | 31.54 | 92.53% | shared-memory scoreboard stalls |
| `norm` | 343.96 GB/s | 76.88% | 68.26% | 0.08% | 3.11% | 31.56 | 20.69 | 86.34% | memory-heavy reduction with predication |
| `quantize_bitpack` | 223.47 GB/s | 56.24% | 45.49% | 56.87% | 64.60% | 21.24 | 17.87 | 88.60% | predication/branch divergence likely dominates |

Interpretation:

```text
quantize_bitpack is the largest encode phase, but it is not saturating memory.
The low not-predicated thread count points at Lloyd-Max bucket control flow.
```

The previous binary-search bucket lookup was slower. The next isolated
microbenchmark should test a different idea: a branchless TQ4 bucket lookup
that keeps one thread per coordinate and still uses the existing bitpack path.
Only if that wins should it be considered for the main TQ path.

## Branchless TQ4 Bucket Microbenchmark

The benchmark now also reports:

```text
quantize_branchless4_alt
```

This variant keeps the same one-thread-per-coordinate bitpack path but replaces
the early-return Lloyd-Max bucket search with a fixed branchless 15-comparison
TQ4 lookup. It is intended to test the Nsight Compute finding that
`quantize_bitpack` has low not-predicated threads per warp.

Run:

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_tq_kernel_microbench
sbatch run_tq_kernel_microbench.slurm
```

Compare:

```text
quantize_bitpack
quantize_branchless4_alt
```

Only move the branchless lookup into the main path if it clearly beats the
current `quantize_bitpack` timing.

## Result: Branchless TQ4 Bucket Microbenchmark

Run:

```text
job_id = 942930
node = gn1220
GPU = 1 x V100
bits = 4
warmup = 20
repeat = 200
```

| rows | cols | quantize_bitpack | quantize_pack4_alt | quantize_branchless4_alt | Branchless speedup |
|---:|---:|---:|---:|---:|---:|
| 256 | 16384 | 0.0989 ms | 0.0996 ms | 0.0689 ms | 30.3% |
| 256 | 32768 | 0.1680 ms | 0.1675 ms | 0.1196 ms | 28.8% |

The branchless TQ4 bucket lookup clearly wins while preserving the existing
one-thread-per-coordinate bitpack structure. This matches the Nsight Compute
signal: `quantize_bitpack` was not memory-saturated, but had low
not-predicated threads per warp.

Decision: **move branchless TQ4 lookup into the main TQ quantize path for
`bits == 4`**. Keep `pack4_alt` as a negative microbenchmark result only.

## Result: Branchless TQ4 in Main Quantize Path

After moving the branchless TQ4 bucket lookup into
`column_tq_lloyd_quantize_kernel`, the normal `quantize_bitpack` phase now
uses the faster branchless lookup for `bits == 4`.

Run:

```text
GPU = 1 x V100
bits = 4
warmup = 20
repeat = 200
```

| rows | cols | encode | decode_add | roundtrip | norm | transform | quantize_bitpack | decode_event | relative error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 16384 | 0.2238 ms | 0.0945 ms | 0.3184 ms | 0.0562 ms | 0.0750 ms | 0.0717 ms | 0.0846 ms | 0.09717795 |
| 256 | 32768 | 0.3782 ms | 0.1606 ms | 0.5388 ms | 0.0955 ms | 0.1330 ms | 0.1242 ms | 0.1500 ms | 0.09720906 |

Compared with the pre-branchless phase breakdown:

| rows | cols | quantize before | quantize after | Change | encode total before | encode total after | Change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 16384 | 0.0987 ms | 0.0717 ms | -27.4% | 0.2368 ms | 0.2102 ms | -11.2% |
| 256 | 32768 | 0.1694 ms | 0.1242 ms | -26.7% | 0.4059 ms | 0.3619 ms | -10.8% |

This confirms that the main TQ4 path now gets nearly the same benefit as the
microbenchmark-only `quantize_branchless4_alt` path. Accuracy is unchanged.

The new largest per-phase costs are now `decode_event`, `transform`, and
`quantize_bitpack`, all in the same rough range. Future TQ-only CUDA work
should therefore target transform/decode memory movement or larger structural
changes, not another small bitpack-only variant.

## Result: Branchless Pair-Pack TQ4 Microbenchmark

The earlier `quantize_pack4_alt` result was negative, but it used the old
branchy Lloyd-Max bucket lookup. Since branchless TQ4 bucket lookup is now
known to help, this microbenchmark adds one more isolated variant:

```text
quantize_pack4_branchless_alt
```

This variant combines:

```text
two TQ4 codes per thread
direct one-byte store
no atomic bitpack write
branchless Lloyd-Max bucket lookup
```

It is still microbenchmark-only. It should not be moved into the main path
unless it clearly beats the current `quantize_bitpack` phase.

Run:

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_tq_kernel_microbench
sbatch run_tq_kernel_microbench.slurm
```

Compare:

```text
quantize_bitpack
quantize_branchless4_alt
quantize_pack4_branchless_alt
```

Run:

```text
GPU = 1 x V100
bits = 4
warmup = 20
repeat = 200
```

| rows | cols | quantize_bitpack | quantize_branchless4_alt | quantize_pack4_branchless_alt |
|---:|---:|---:|---:|---:|
| 256 | 16384 | 0.0723 ms | 0.0695 ms | 0.0462 ms |
| 256 | 32768 | 0.1249 ms | 0.1222 ms | 0.0777 ms |

The branchless pair-pack variant clearly beats the current main TQ4 quantize
phase:

| rows | cols | Main quantize | Branchless pair-pack | Change |
|---:|---:|---:|---:|---:|
| 256 | 16384 | 0.0723 ms | 0.0462 ms | -36.1% |
| 256 | 32768 | 0.1249 ms | 0.0777 ms | -36.2% |

This result is different from the older `quantize_pack4_alt` negative result
because the older pair-pack variant still used the branchy Lloyd-Max bucket
lookup. Combining pair-pack with branchless TQ4 lookup keeps the no-atomic
byte-store benefit without reintroducing bucket divergence.

Decision after microbenchmark: **try moving branchless pair-pack into the main
TQ4 encode path**. The trial should also skip `clear_codes` for TQ4 because the
pair-pack kernel writes each output byte directly instead of using `atomicOr`.

Validation plan after moving it into the main path:

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_tq_kernel_microbench
sbatch run_tq_kernel_microbench.slurm
```

Expected: `clear_codes` should be near zero for TQ4, and `quantize_bitpack`
should be close to the `quantize_pack4_branchless_alt` numbers above. If that
passes, run the full RSVD validation in `../exp_cuda_opt_round1`.

Full RSVD result: **fallback**. The microbenchmark path succeeded, but the
full 8-GPU RSVD timing regressed from `116.201 ms` to `124.048 ms`. The main
TQ4 path was restored to the previous branchless one-thread-per-coordinate
quantize kernel with generic atomic bitpack. Keep `quantize_pack4_branchless_alt`
only as a microbenchmark/diagnostic result.

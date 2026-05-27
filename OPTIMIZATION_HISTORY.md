# TurboQuant Multi-GPU SVD Optimization History

This note records the project-level optimization path for TurboQuant/QJL
accelerated randomized SVD. It started as the `randomized_svd_baseline_v2`
single-node experiment log, but now also covers shared `turboquant/` kernels
and the `randomized_svd_baseline_v3` multi-node compressed MPI collective path.

It is written as an experiment log: each section explains what was changed, why
it was tried, what data was collected, and what conclusion we kept.

## Table of Content

- [TurboQuant Multi-GPU SVD Optimization History](#turboquant-multi-gpu-svd-optimization-history)
  - [Table of Content](#table-of-content)
  - [Current CUDA Implementation](#current-cuda-implementation)
  - [1. Build a Complete Multi-GPU Randomized SVD Baseline](#1-build-a-complete-multi-gpu-randomized-svd-baseline)
    - [Change](#change)
    - [Reason](#reason)
    - [Result](#result)
    - [Conclusion](#conclusion)
  - [2. Add a Simple Low-Bit Quantization Baseline](#2-add-a-simple-low-bit-quantization-baseline)
    - [Change](#change-1)
    - [Reason](#reason-1)
    - [Representative Result](#representative-result)
    - [Conclusion](#conclusion-1)
  - [3. Move Random Input Generation to Device](#3-move-random-input-generation-to-device)
    - [Change](#change-2)
    - [Reason](#reason-2)
    - [Result](#result-1)
    - [Conclusion](#conclusion-2)
  - [4. Remove Host Reconstruction From Timing](#4-remove-host-reconstruction-from-timing)
    - [Change](#change-3)
    - [Reason](#reason-3)
    - [Result](#result-2)
    - [Conclusion](#conclusion-3)
  - [5. Add Repeat Timing and Exclude Cold Start](#5-add-repeat-timing-and-exclude-cold-start)
    - [Change](#change-4)
    - [Reason](#reason-4)
    - [Representative Result](#representative-result-1)
    - [Conclusion](#conclusion-4)
  - [6. Implement TurboQuant on `B_i`](#6-implement-turboquant-on-b_i)
    - [Change](#change-5)
    - [Reason](#reason-5)
    - [Main Result](#main-result)
    - [Conclusion](#conclusion-5)
  - [7. Try TQ + QJL Residual Correction](#7-try-tq--qjl-residual-correction)
    - [Change](#change-6)
    - [Reason](#reason-6)
    - [Timing Result](#timing-result)
    - [Error Result: Alpha Grid](#error-result-alpha-grid)
    - [Error Result: QJL Dimension Equal to Vector Dimension](#error-result-qjl-dimension-equal-to-vector-dimension)
    - [Conclusion](#conclusion-6)
  - [8. Replace Hash Signs With Gaussian QJL Samples](#8-replace-hash-signs-with-gaussian-qjl-samples)
    - [Change](#change-7)
    - [Reason](#reason-7)
    - [Result](#result-3)
    - [Conclusion](#conclusion-7)
  - [9. Profile With Nsight Systems and Nsight Compute](#9-profile-with-nsight-systems-and-nsight-compute)
    - [Change](#change-8)
    - [Reason](#reason-8)
    - [Nsight Systems Observation](#nsight-systems-observation)
    - [Conclusion](#conclusion-8)
  - [10. Avoid Optimizations That Only Improve the Shared Baseline](#10-avoid-optimizations-that-only-improve-the-shared-baseline)
    - [Change](#change-9)
    - [Reason for Backing Off](#reason-for-backing-off)
    - [Policy Kept](#policy-kept)
    - [Conclusion](#conclusion-9)
  - [11. Move From v2 to v3 MPI Compressed Collectives](#11-move-from-v2-to-v3-mpi-compressed-collectives)
    - [Change](#change-10)
    - [Reason](#reason-9)
    - [Result](#result-4)
    - [GPU Scale Limit](#gpu-scale-limit)
    - [Conclusion](#conclusion-10)
  - [12. Optimize Only TQ-Specific Kernels](#12-optimize-only-tq-specific-kernels)
    - [Policy](#policy)
    - [12.1 Remove TQ-Only Tail Synchronization](#121-remove-tq-only-tail-synchronization)
      - [Change](#change-11)
      - [Result](#result-5)
      - [Conclusion](#conclusion-11)
    - [12.2 Fuse TQ Decode-Add](#122-fuse-tq-decode-add)
      - [Change](#change-12)
      - [Result](#result-6)
      - [Conclusion](#conclusion-12)
  - [Current Best Result](#current-best-result)
  - [Recommended Next Experiments](#recommended-next-experiments)

## Current CUDA Implementation

The main executables are:

```text
randomized_svd_baseline_v2/randomized_svd_multigpu_v2   single-node, multi-GPU
randomized_svd_baseline_v3/randomized_svd_multigpu_v3   multi-node MPI + compressed collectives
```

The main implementation lives in:

```text
randomized_svd_baseline_v2/randomized_svd_multigpu_v2.cu
randomized_svd_baseline_v3/randomized_svd_multigpu_v3.cu
turboquant/turboquant.hpp
turboquant/turboquant.cu
```

At a high level, the `randomized_svd_baseline_*` code owns the randomized SVD
pipeline, GPU memory layout, cuBLAS/cuSOLVER calls, Slurm benchmark options,
MPI communication, and timing/reporting. The `turboquant/` code owns the
compression kernels used when reducing the distributed `B_i` matrices.

The current multi-GPU data layout is row-distributed:

```text
GPU i owns A_i
A = [A_0; A_1; ...; A_{g-1}]
```

v2 uses one process on one node. v3 uses one MPI rank per node, and each rank
controls the visible local GPUs:

```text
rank 0 -> node 0, local GPUs 0..7
rank 1 -> node 1, local GPUs 0..7
global GPUs = mpi_size * local_gpus_per_rank
```

The main GPU-side SVD pipeline is:

```text
1. Generate or upload A_i on each GPU.
2. Compute Y_i = A_i * Omega with cuBLAS.
3. Compute local QR Y_i = Qbar_i * R_i with cuSOLVER.
4. Gather small R_i factors and run TSQR reduction on GPU 0.
5. Broadcast/apply TSQR transform T_i to form Q_i.
6. Compute B_i = Q_i^T * A_i with cuBLAS.
7. Optionally compress each B_i.
8. Reconstruct B_i_hat on GPU and reduce B = sum_i B_i_hat on GPU 0.
9. Compute SVD(B) on GPU 0 with cuSOLVER.
10. Optionally form U_i = Q_i * U_tilde.
```

v3 adds cross-node communication around the same basic pipeline:

```text
1. Each rank gathers local R_i factors.
2. MPI rank 0 runs TSQR on stacked R_i and broadcasts T_i.
3. Each rank forms a node-local B.
4. For TQ/lowbit/TQ+QJL, each rank compresses node-local B.
5. MPI gathers compressed B payloads and metadata.
6. Rank 0 decodes/adds compressed payloads on GPU 0.
7. Rank 0 computes SVD(B).
```

The currently implemented `B_i` compression modes are:

| Mode | Option | Meaning |
| --- | --- | --- |
| none | `--compress-b-mode none` | Use FP32 `B_i` directly. |
| lowbit | `--compress-b-mode lowbit` | Direct scalar low-bit quantization. |
| TQ | `--compress-b-mode tq` | Random sign + FWHT rotation + low-bit quantization + inverse rotation. |
| TQ+QJL | `--compress-b-mode tq-qjl` | TQ plus experimental QJL residual correction. |

The main TurboQuant kernels currently do the following on GPU:

```text
apply random signs
pad/truncate vectors for FWHT
run normalized FWHT
quantize/dequantize 8-bit, 4-bit, or 2-bit payloads
run inverse normalized FWHT
apply inverse signs
produce reconstructed B_i_hat for GPU-side reduction
```

The current TQ path uses column-wise FWHT on the short `l` dimension of `B`
(`l = k + oversample`). The optimized decode-add path fuses packed-code
dequantization, inverse FWHT, inverse signs, and accumulation into one shared
memory kernel when `padded_rows <= 1024`.

The TQ+QJL path additionally does:

```text
form residual = B_i - B_i_hat_TQ
generate deterministic Gaussian sketch samples
compute QJL residual sketch
reconstruct an approximate residual correction
add alpha * residual_hat to B_i_hat_TQ
```

Important implementation choices:

```text
--device-random-input  keeps benchmark input generation on GPU.
--skip-form-u          skips final U_i formation for timing-focused runs.
--no-check-error       avoids full A_hat reconstruction.
--no-check-b-error     avoids copying B_i/B_i_hat back only for error checks.
--repeat N             reports warm timing after excluding the first run.
```

For fair timing, the main scripts use device-random input, skip host-side full
reconstruction, and compare `none`, `tq`, and `tq-qjl` under the same SVD
pipeline.

Unless stated otherwise, the main comparison setting is:

```text
m=32768, n=8192, k=256, oversample=64, l=320, ngpus=2
GPU: 2x Tesla V100-SXM2-32GB
input: --device-random-input
timing: --repeat 100 or larger, cold start excluded
U formation: --skip-form-u
B reduction: host_reduce_b=no
```

The current v3 multi-node comparison setting is:

```text
m=65536, n=16384, k=256, oversample=64, l=320
nodes=2, ranks=2, gpus_per_rank=8, global_gpus=16
GPU: 16x Tesla V100-SXM2-32GB
input: --device-random-input
timing: --repeat 20, cold start excluded
U formation: --skip-form-u
MPI B collective: fp32_reduce for none, compressed_gather_decode for TQ
```

The main timing metric is `warm_pipeline_avg_ms`, which excludes the first
iteration. This avoids counting CUDA/cuBLAS/cuSOLVER initialization cost.

For compression quality, the main reported metric is
`reduce_B_relative_error`, meaning:

```text
||B - B_hat||_F / ||B||_F
```

This measures the compression error on the distributed `B_i` reduction target.
It is not the full `||A - U S V^T||_F / ||A||_F` SVD reconstruction error.

## 1. Build a Complete Multi-GPU Randomized SVD Baseline

### Change

The original baseline was partial multi-GPU. The v2 path was changed into a
TSQR-style distributed randomized SVD:

```text
A_i on GPU i
Y_i = A_i * Omega
Y_i = Qbar_i * R_i
TSQR reduce R_i
Q_i = Qbar_i * T_i
B_i = Q_i^T * A_i
B = sum_i B_i
SVD(B)
U_i = Q_i * U_tilde
```

### Reason

This gives a real multi-GPU communication target. In v2, the most important
compressible tensor is `B_i`, not only the earlier `Y_i` gather.

### Result

For the main size:

| Tensor | FP32 Payload |
| --- | ---: |
| TSQR `R_i` gather | 0.78125 MiB |
| `B_i` reduction | 20 MiB |

### Conclusion

Compressing `R_i` is too small to be the main story. Compressing `B_i` is the
right target because it is much larger and grows with GPU count:

```text
B_i payload = ngpus * l * n * sizeof(float)
```

## 2. Add a Simple Low-Bit Quantization Baseline

### Change

Before implementing TurboQuant, a simple scalar quantization path was added:

```text
scale = max(abs(x)) / qmax
q = clamp(round(x / scale), -qmax, qmax)
x_hat = q * scale
```

This was later renamed conceptually as "low-bit quantization", not
TurboQuant.

### Reason

This gives a minimal baseline for measuring how much compression is possible
without random rotation or QJL residual correction.

### Representative Result

Source: `randomized_svd_multigpu_v2_921643.out`

Small setting:

```text
m=4096, n=2048, k=64, l=80, ngpus=2
```

| Mode | Payload | Ratio | Compress Time | Full SVD Error |
| --- | ---: | ---: | ---: | ---: |
| none | 1.25 MiB | 1.00x | 1.25788 ms | 1.02421 |
| lowbit 8-bit | 0.312508 MiB | 3.9999x | 3.27215 ms | 1.02263 |
| lowbit 4-bit | 0.156258 MiB | 7.99961x | 2.01179 ms | 1.02293 |

Medium setting:

```text
m=8192, n=4096, k=128, l=160, ngpus=2
```

| Mode | Payload | Ratio | Compress Time | Full SVD Error |
| --- | ---: | ---: | ---: | ---: |
| none | 5 MiB | 1.00x | 4.46356 ms | 1.02405 |
| lowbit 8-bit | 1.25001 MiB | 3.99998x | 3.72091 ms | 1.02357 |
| lowbit 4-bit | 0.625008 MiB | 7.9999x | 4.23288 ms | 1.02306 |

### Conclusion

The low-bit path validated the compression pipeline, but it is not the final
method. It also showed that full SVD error is too coarse for judging `B_i`
compression quality, so later experiments use `reduce_B_relative_error`.

## 3. Move Random Input Generation to Device

### Change

Added `--device-random-input` to generate the test matrix directly on GPU.

### Reason

The earlier path created large random matrices on host and copied them to GPUs.
That made benchmark timing noisy and inflated by host setup.

### Result

The main benchmark now reports:

```text
init_host_random          ~0 ms
```

when `--device-random-input` is enabled.

### Conclusion

This should stay enabled for timing runs. It makes the benchmark focus on the
randomized SVD pipeline and compression kernels instead of host data setup.

## 4. Remove Host Reconstruction From Timing

### Change

Avoided host-side `A_hat = U S V^T` reconstruction for timing. Added options:

```text
--skip-form-u
--no-check-error
--no-check-b-error
```

### Reason

The project goal is to compare GPU SVD pipeline time and compression overhead.
Building a full `A_hat` on host is not part of the GPU algorithm and can dwarf
the kernel time.

### Result

The main timing scripts now measure:

```text
Y_i projection
QR / TSQR
Q_i formation
B_i build + compressed reduction
SVD(B)
```

and do not include full host reconstruction.

### Conclusion

This is the correct timing policy for method comparison. Full SVD error can be
checked in separate accuracy jobs, but should not be in the hot timing path.

## 5. Add Repeat Timing and Exclude Cold Start

### Change

Added:

```text
--repeat N
--repeat-print-every K
```

The first iteration is reported as `cold_pipeline_ms`; the average excludes it:

```text
warm_compute_avg_ms
warm_pipeline_avg_ms
```

### Reason

The first run includes CUDA context creation, handle creation, memory setup,
and library warmup. It is not representative of steady-state kernel time.

### Representative Result

Source: `randomized_svd_multigpu_v2_tq_bit_curve_925952.out`

| Mode | Cold Pipeline | Warm Pipeline Avg |
| --- | ---: | ---: |
| none | 301.548 ms | 50.2152 ms |
| TQ 4-bit | 295.383 ms | 50.1071 ms |
| TQ 2-bit | not used as final | 49.7375 ms |

### Conclusion

All final timing comparisons should use warm averages. Cold-start numbers are
useful only for diagnosing setup overhead.

## 6. Implement TurboQuant on `B_i`

### Change

Added `--compress-b-mode tq`, which applies:

```text
random sign
normalized FWHT rotation
low-bit quantization in rotated domain
inverse FWHT
inverse sign
```

The implementation is in `turboquant/` and is called from
`randomized_svd_multigpu_v2.cu`.

### Reason

TurboQuant should make the value distribution more quantization-friendly than
direct scalar quantization, especially at low bit widths.

### Main Result

Source: `randomized_svd_multigpu_v2_tq_bit_curve_925952.out`

```text
m=32768, n=8192, k=256, l=320, ngpus=2, repeat=100
```

| Mode | Payload | Ratio | Warm Pipeline Avg | B Error |
| --- | ---: | ---: | ---: | ---: |
| none | 20 MiB | 1.00x | 50.2152 ms | skipped |
| TQ 4-bit | 4.00002 MiB | 4.99998x | 50.1071 ms | 0.171443 |
| TQ 2-bit | 2.00002 MiB | 9.99992x | 49.7375 ms | 0.948631 |

Representative steady-state compression times from the same run:

| Mode | Compress Time |
| --- | ---: |
| none | ~0.00005 ms |
| TQ 4-bit | ~1.18 ms |
| TQ 2-bit | ~1.08 ms |

### Conclusion

TQ 4-bit is the best current headline result:

```text
~5x B_i payload compression
similar or slightly lower warm pipeline time than none
moderate B compression error: 0.171443
```

TQ 2-bit gives ~10x compression but has much larger B error, so it should be
presented as an aggressive compression point rather than the main setting.

## 7. Try TQ + QJL Residual Correction

### Change

Added `--compress-b-mode tq-qjl`.

The idea was:

```text
x_hat_TQ = TQ(x)
residual = x - x_hat_TQ
sketch residual with QJL
reconstruct approximate residual
x_hat = x_hat_TQ + alpha * residual_hat
```

### Reason

The expectation was that QJL residual information might allow lower bits at
the same error, especially for TQ 2-bit.

### Timing Result

Source: `randomized_svd_multigpu_v2_tq_focus_925805.out`

```text
m=32768, n=8192, k=256, l=320, ngpus=2, repeat=200
qjl_dim=64, qjl_alpha=0.01
```

| Mode | Payload | Ratio | Warm Pipeline Avg |
| --- | ---: | ---: | ---: |
| none | 20 MiB | 1.00x | 50.2974 ms |
| TQ 4-bit | 4.00002 MiB | 4.99998x | 49.9718 ms |
| TQ 2-bit | 2.00002 MiB | 9.99992x | 49.8283 ms |
| TQ+QJL 4-bit | 4.0005 MiB | 4.99937x | 52.952 ms |
| TQ+QJL 2-bit | 2.0005 MiB | 9.99748x | 52.9552 ms |

### Error Result: Alpha Grid

Source: `randomized_svd_multigpu_v2_qjl_alpha_grid_925949.out`

```text
qjl_dim=16, bits=4
```

| Mode | Alpha | B Error | Compress Time |
| --- | ---: | ---: | ---: |
| TQ 4-bit | 0 | 0.171443 | 3.63762 ms |
| TQ+QJL | -1.0 | 86.971361 | 6.14777 ms |
| TQ+QJL | -0.5 | 43.486154 | 6.11591 ms |
| TQ+QJL | -0.25 | 21.743804 | 6.08315 ms |
| TQ+QJL | -0.1 | 8.699206 | 5.87556 ms |
| TQ+QJL | -0.05 | 4.352356 | 5.90839 ms |
| TQ+QJL | -0.01 | 0.886877 | 5.88429 ms |
| TQ+QJL | 0.0 | 0.171443 | 2.65269 ms |

### Error Result: QJL Dimension Equal to Vector Dimension

Source: `randomized_svd_multigpu_v2_qjl_dim_d_925950.out`

```text
qjl_dim=320 = l, bits=4
```

| Mode | Alpha | B Error | Compress Time |
| --- | ---: | ---: | ---: |
| TQ 4-bit | 0 | 0.171443 | 2.61949 ms |
| TQ+QJL | -1.0 | 19.459571 | 35.53 ms |
| TQ+QJL | -0.1 | 1.954814 | 32.4537 ms |
| TQ+QJL | -0.01 | 0.260493 | 31.8564 ms |
| TQ+QJL | -0.001 | 0.172719 | 31.7662 ms |
| TQ+QJL | 0.0 | 0.171443 | 2.62227 ms |
| TQ+QJL | 0.001 | 0.172368 | 31.7862 ms |
| TQ+QJL | 0.01 | 0.258158 | 31.7637 ms |
| TQ+QJL | 0.1 | 1.951714 | 31.7385 ms |
| TQ+QJL | 1.0 | 19.456460 | 32.0607 ms |

### Conclusion

Current TQ+QJL is not better than TQ alone.

Observed behavior:

```text
alpha = 0 reproduces TQ error
any nonzero alpha worsens B error in tested grids
larger qjl_dim greatly increases compress time
```

The current QJL residual reconstruction should be treated as an exploratory
negative result, not as the final accelerated method.

## 8. Replace Hash Signs With Gaussian QJL Samples

### Change

The QJL sketch first used hash/Rademacher signs. It was changed to a
deterministic Gaussian generator based on hash + Box-Muller.

### Reason

The paper formulation uses Gaussian sketching. The hash sign version was a
fast approximation, but it was not the same distribution.

### Result

Even with Gaussian QJL samples and `qjl_dim = d`, QJL did not improve error.
The best result remained `alpha = 0`, which is equivalent to no residual
correction.

### Conclusion

The issue is not only the sign distribution. The residual reconstruction model
itself is likely too crude for this use case.

## 9. Profile With Nsight Systems and Nsight Compute

### Change

Added profiling scripts:

```text
run_randomized_svd_multigpu_v2_nsys.slurm
run_randomized_svd_multigpu_v2_ncu.slurm
```

### Reason

The goal was to separate:

```text
whole SVD pipeline time
compression-only time
library calls
custom CUDA kernels
host/device copies
```

### Nsight Systems Observation

Source: `randomized_svd_multigpu_v2_nsys_925490.out`

For one profiled TQ 4-bit run:

| Region | Time |
| --- | ---: |
| local_projection_Yi | 255.161 ms |
| svd_B_on_gpu0 | 71.231 ms |
| local_qr_Yi | 38.452 ms |
| build_reduce_Bi | 20.302 ms |
| compress_Bi | 9.522 ms |
| tsqr_R_reduce_gpu0 | 5.465 ms |

CUDA memory traffic summary:

| Direction | Total |
| --- | ---: |
| Host-to-Device | 13.728 MB |
| Device-to-Host | 16.322 MB |
| Device-to-Device | 0.328 MB |

### Conclusion

For optimizing TQ/TQ+QJL specifically, focus on `compress_Bi` and avoid
optimizations that mainly speed up the shared `none` path. The larger pipeline
is still dominated by projection, QR, and SVD(B), so compression improvements
must be measured carefully with repeat timing.

## 10. Avoid Optimizations That Only Improve the Shared Baseline

### Change

Some experiments targeted common work shared by `none`, `tq`, and `tq-qjl`,
for example changing B reduction behavior globally.

### Reason for Backing Off

The project goal is to make TQ/TQ+QJL better relative to `none`. If an
optimization speeds up all modes equally, it may improve the program but does
not demonstrate compression-specific acceleration.

### Policy Kept

Use the same common pipeline for all modes:

```text
host_reduce_b=no
device_random_input=yes
skip_form_u=yes
repeat timing
```

Then only compare changes inside the compression path.

### Conclusion

This keeps the comparison fair:

```text
none measures the uncompressed B_i reduction
tq measures the same pipeline plus TurboQuant compression
tq-qjl measures the same pipeline plus TurboQuant and QJL residual correction
```

## 11. Move From v2 to v3 MPI Compressed Collectives

### Change

Added `randomized_svd_baseline_v3`, a multi-node version using one MPI rank per
node and all visible GPUs per rank. The important v3 communication change is
that cross-node `B` reduction no longer has to send FP32 `B` in compressed
modes.

For `--compress-b-mode tq`, `lowbit`, or `tq-qjl`, v3 now does:

```text
rank-local B on GPU0
compress rank-local B
MPI gather compressed codes + scale + residual metadata
rank 0 decodes/adds each payload on GPU0
rank 0 computes SVD(B)
```

For `--compress-b-mode none`, v3 still uses FP32 `MPI_Reduce`.

### Reason

Earlier v3 experiments compressed only the node-local per-GPU `B_i` payload,
but the cross-node MPI reduction still sent FP32 `B`. That validated multi-node
correctness, but it was not yet a final compressed-collective story.

### Result

16-GPU base run:

```text
job_id=927251
m=32768 n=8192 k=256 l=320
nodes=2 ranks=2 gpus_per_rank=8 global_gpus=16
repeat=5, cold start excluded
```

| Mode | Warm pipeline avg | Node-local B payload | MPI B payload | MPI B collective | B relative error |
| --- | ---: | ---: | ---: | --- | ---: |
| none | 64.5616 ms | 160 MiB | 20 MiB | fp32_reduce | skipped |
| tq 4-bit | 50.0871 ms | 32.0001 MiB | 4.00002 MiB | compressed_gather_decode | 0.175305 |
| tq 2-bit | 47.1681 ms | 16.0001 MiB | 2.00002 MiB | compressed_gather_decode | 0.952533 |

16-GPU larger matrix run:

```text
job_id=927252
m=65536 n=16384 k=256 l=320
nodes=2 ranks=2 gpus_per_rank=8 global_gpus=16
repeat=20, cold start excluded
```

| Mode | Warm pipeline avg | Node-local B payload | MPI B payload | MPI B collective | B relative error |
| --- | ---: | ---: | ---: | --- | ---: |
| none | 90.4434 ms | 320 MiB | 40 MiB | fp32_reduce | skipped |
| tq 4-bit | 73.2406 ms | 64.0001 MiB | 8.00002 MiB | compressed_gather_decode | 0.180200 |
| tq 2-bit | 66.5009 ms | 32.0001 MiB | 4.00002 MiB | compressed_gather_decode | 0.960169 |

### GPU Scale Limit

A 4-node / 32-GPU probe was submitted:

```text
job_id=927250
request=4 nodes x 8 GPUs = 32 GPUs
result=PENDING (QOSMaxNodePerJobLimit)
```

So the current practical maximum for one Slurm job is:

```text
2 nodes x 8 GPUs = 16 GPUs
```

### Conclusion

v3 is now the better place to report final multi-node claims. It demonstrates
compressed cross-node payloads, while v2 remains useful for single-node kernel
profiling and simpler debugging.

## 12. Optimize Only TQ-Specific Kernels

### Policy

After comparing `none`, `tq`, and `tq-qjl`, we should avoid optimizing common
pipeline stages when the goal is to improve TQ relative to `none`.

Do not use TQ-performance claims from changes that mainly affect:

```text
Y = A Omega
local QR
TSQR
Q_i formation
B_i = Q_i^T A_i
SVD(B)
```

Allowed TQ-specific optimization targets:

```text
compress_Bi_payload
TQ FWHT kernels
low-bit pack/unpack kernels
TQ decode-add kernels
v3 compressed MPI B payload handling
QJL residual sketch kernels
```

### 12.1 Remove TQ-Only Tail Synchronization

#### Change

Removed the final `cudaStreamSynchronize` from the TQ-only column quantization
path when the caller provides a persistent external scratch buffer. The QJL path
keeps synchronization because it allocates/free temporary residual buffers.

#### Result

Large v3 run:

```text
job_id=927253
m=65536 n=16384 k=256 l=320
nodes=2 ranks=2 gpus_per_rank=8 global_gpus=16
repeat=20
```

| Mode | Before | After | Change |
| --- | ---: | ---: | ---: |
| none | 90.4434 ms | 90.4659 ms | +0.02 ms |
| tq 4-bit | 73.2406 ms | 72.1659 ms | -1.07 ms |
| tq 2-bit | 66.5009 ms | 66.2885 ms | -0.21 ms |

#### Conclusion

This is a clean TQ-only improvement: `none` is effectively unchanged, while TQ
gets a small speedup by allowing more overlap across GPUs.

### 12.2 Fuse TQ Decode-Add

#### Change

The column TQ decode-add path used to do:

```text
packed codes -> dequantize into global d_work
d_work -> inverse FWHT -> inverse signs -> add to B
```

It now uses a fused shared-memory kernel for `padded_rows <= 1024`:

```text
packed codes -> shared-memory dequantize -> inverse FWHT -> inverse signs -> add to B
```

This removes one global memory round trip and one kernel launch from the common
TQ decode path.

#### Result

Large v3 run:

```text
job_id=927254
m=65536 n=16384 k=256 l=320
nodes=2 ranks=2 gpus_per_rank=8 global_gpus=16
repeat=20
```

| Mode | After sync removal | After fused decode-add | Change |
| --- | ---: | ---: | ---: |
| none | 90.4659 ms | 90.2220 ms | -0.24 ms |
| tq 4-bit | 72.1659 ms | 71.4897 ms | -0.68 ms |
| tq 2-bit | 66.2885 ms | 64.8231 ms | -1.47 ms |

The measured B errors stayed the same:

| Mode | B relative error |
| --- | ---: |
| tq 4-bit | 0.180200 |
| tq 2-bit | 0.960169 |

#### Conclusion

This is another TQ-specific improvement. It benefits 2-bit more than 4-bit,
which is consistent with decode/pack overhead becoming more visible at lower
payload sizes.

## Current Best Result

The best current multi-node result to present is:

```text
Method: v3 TQ 4-bit on B / compressed MPI B collective
Matrix: m=65536, n=16384, k=256, l=320
GPUs: 16x V100 across 2 nodes
MPI B payload: 40 MiB -> 8.00002 MiB
Compression ratio: 4.99999x
Warm pipeline avg: 71.4897 ms vs none 90.2220 ms
B relative error: 0.180200
```

The aggressive multi-node setting is:

```text
Method: v3 TQ 2-bit on B / compressed MPI B collective
MPI B payload: 40 MiB -> 4.00002 MiB
Compression ratio: 9.99996x
Warm pipeline avg: 64.8231 ms vs none 90.2220 ms
B relative error: 0.960169
```

The best single-node v2 reference result is:

```text
Method: TQ 4-bit on B_i
Matrix: m=32768, n=8192, k=256, l=320
GPUs: 2x V100
Payload: 20 MiB -> 4.00002 MiB
Compression ratio: 4.99998x
Warm pipeline avg: 50.1071 ms vs none 50.2152 ms
B relative error: 0.171443
```

The aggressive setting is:

```text
Method: TQ 2-bit on B_i
Payload: 20 MiB -> 2.00002 MiB
Compression ratio: 9.99992x
Warm pipeline avg: 49.7375 ms
B relative error: 0.948631
```

The negative result is:

```text
TQ+QJL currently does not reduce error at the same bit width.
Nonzero alpha worsens B error in tested grids.
Large qjl_dim increases compression time substantially.
```

## Recommended Next Experiments

1. Keep the main performance claim on v3 TQ 4-bit compressed MPI collectives.
2. Use v3 TQ 2-bit as the aggressive compression point, with clear error caveat.
3. Keep QJL as an exploratory negative result unless the estimator is redesigned.
4. If QoS allows more nodes later, rerun the same v3 comparison above 16 GPUs.
5. If optimizing further, profile only TQ/TQ+QJL-specific kernels and compare
   against the same uncompressed pipeline. Do not claim improvements from
   changes that also speed up `none`.

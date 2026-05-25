# Randomized SVD v2 Optimization History

This note records the optimization path for `randomized_svd_baseline_v2`.
It is written as an experiment log: each section explains what was changed,
why it was tried, what data was collected, and what conclusion we kept.

Unless stated otherwise, the main comparison setting is:

```text
m=32768, n=8192, k=256, oversample=64, l=320, ngpus=2
GPU: 2x Tesla V100-SXM2-32GB
input: --device-random-input
timing: --repeat 100 or larger, cold start excluded
U formation: --skip-form-u
B reduction: host_reduce_b=no
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

## Current Best Result

The best current result to present is:

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

1. Keep the main claim on TQ 4-bit.
2. Use TQ 2-bit as an aggressive compression point.
3. Keep QJL as an exploratory result unless the estimator is redesigned.
4. Run the same TQ 4-bit and none comparison on more GPUs once QoS allows it.
5. If optimizing further, profile only the compression kernels and compare
   against the same uncompressed pipeline.


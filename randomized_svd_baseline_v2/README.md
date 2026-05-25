# Randomized SVD Baseline v2

This version is a TSQR-style multi-GPU randomized SVD baseline.

Compared with `randomized_svd_baseline`, v2 keeps the large matrix blocks on
their owning GPUs after the first projection:

1. Split `A` by rows: GPU `i` owns `A_i`.
2. Compute `Y_i = A_i * Omega` on each GPU.
3. Compute local QR: `Y_i = Qbar_i * R_i`.
4. Gather the small `R_i` factors and run a TSQR reduction.
5. Form distributed `Q_i = Qbar_i * T_i`.
6. Compute local `B_i = Q_i^T * A_i` on each GPU.
7. Reduce `B = sum_i B_i`.
8. Compute SVD of small `B` on GPU 0.
9. Broadcast `U_tilde_k` and form distributed `U_i = Q_i * U_tilde_k`.

The current implementation still uses a host-mediated `R_i` TSQR gather for
portability. The `B_i` reduction path keeps reconstructed blocks on device and
accumulates `B = sum_i B_i_hat` on GPU 0 before running `SVD(B)`.

## TurboQuant/QJL Compression Targets

There are two randomized SVD baselines in this repository:

- `randomized_svd_baseline`: v1, partial multi-GPU.
- `randomized_svd_baseline_v2`: v2, TSQR-style distributed baseline.

The two versions expose different communication tensors for TurboQuant/QJL.

### Version 1: Compress Gathered `Y_i`

Version 1 computes the local randomized projections on multiple GPUs:

```text
Y_i = A_i * Omega
```

Then it gathers all `Y_i` blocks to host and assembles the full `Y`. This is
the main compression point in v1.

Compressible tensor:

```text
Y_i, or equivalently the gathered Y
```

Payload formula:

```text
m * l * sizeof(float)
```

where `l = k + oversample`.

For the current benchmark configurations:

| Config | Total FP32 Payload | Per-GPU Payload, 2 GPUs | Per-GPU Payload, 8 GPUs |
| --- | ---: | ---: | ---: |
| `m=4096, n=2048, k=64, l=80` | `1.25 MiB` | `0.625 MiB` | `0.156 MiB` |
| `m=8192, n=4096, k=128, l=160` | `5 MiB` | `2.5 MiB` | `0.625 MiB` |

The total payload does not grow with GPU count. More GPUs only split the same
`Y` tensor into smaller row blocks.

### Version 2: Compress `R_i` and `B_i`

Version 2 has two communication points.

First, local QR produces one small triangular factor per GPU:

```text
Y_i = Qbar_i * R_i
```

The `R_i` factors are gathered for the TSQR reduction.

Compressible tensor:

```text
R_i
```

Payload formula:

```text
ngpus * l * l * sizeof(float)
```

For the current benchmark configurations:

| Config | Total FP32 Payload, 2 GPUs | Total FP32 Payload, 8 GPUs |
| --- | ---: | ---: |
| `m=4096, n=2048, k=64, l=80` | `0.0488 MiB` | `0.195 MiB` |
| `m=8192, n=4096, k=128, l=160` | `0.195 MiB` | `0.781 MiB` |

This payload is small, so compressing `R_i` is useful mostly as a secondary
experiment.

Second, each GPU computes its own low-dimensional product:

```text
B_i = Q_i^T * A_i
```

Then the partial matrices are reduced:

```text
B = sum_i B_i
```

This is the primary compression target in v2.

Compressible tensor:

```text
B_i
```

Payload formula:

```text
ngpus * l * n * sizeof(float)
```

For the current benchmark configurations:

| Config | Total FP32 Payload, 2 GPUs | Total FP32 Payload, 8 GPUs | Per-GPU Payload |
| --- | ---: | ---: | ---: |
| `m=4096, n=2048, k=64, l=80` | `1.25 MiB` | `5 MiB` | `0.625 MiB` |
| `m=8192, n=4096, k=128, l=160` | `5 MiB` | `20 MiB` | `2.5 MiB` |

Unlike v1's `Y_i` gather, the total `B_i` reduction payload grows with GPU
count. This makes `B_i` the best target for demonstrating TurboQuant/QJL on an
8-GPU run.

### Recommended Path

For the fastest implementation path:

```text
v1: compress Y_i first
```

This validates the numerical effect of TurboQuant/QJL with the simplest
pipeline.

For the stronger multi-GPU story:

```text
v2: compress B_i reduction
```

This targets a communication point created by the more complete distributed
pipeline. It is the recommended final-project target once v2 is stable.

## Compression Runtime Options

The v2 executable can optionally compress each `B_i` block before the
GPU-side reduction on GPU 0.

```bash
./randomized_svd_multigpu_v2 \
  --m 4096 \
  --n 2048 \
  --k 64 \
  --oversample 16 \
  --ngpus 2 \
  --compress-b-mode lowbit \
  --compress-b-bits 8 \
  --qjl-dim 256 \
  --qjl-alpha 0.25
```

Supported modes:

```text
--compress-b-mode none     FP32 baseline, no compression
--compress-b-mode lowbit   GPU-side low-bit scalar quantization baseline
--compress-b-mode tq       GPU-side TurboQuant rotation + low-bit quantization
--compress-b-mode tq-qjl   GPU-side TurboQuant plus QJL residual sketch
```

For `lowbit`, supported bits are:

```text
--compress-b-bits 8
--compress-b-bits 4
--compress-b-bits 2
```

This is the earlier simple quantization baseline:

```text
scale = max(abs(x)) / qmax
q = clamp(round(x / scale), -qmax, qmax)
x_hat = q * scale
```

For `tq`, supported bits are also `8`, `4`, and `2`. It does:

```text
random sign + Hadamard rotation
low-bit scalar quantization in the rotated domain
inverse Hadamard + inverse sign reconstruction
```

For `tq-qjl`, the current implementation adds:

```text
residual = x - x_hat_TQ
QJL residual sign sketch on GPU
approximate residual reconstruction from the sign sketch on GPU
```

The QJL correction uses a one-bit sketch reconstruction approximation. It is a
prototype for measuring numerical behavior, not yet a fully optimized estimator.

Use `--qjl-alpha` to control residual correction strength:

```text
--qjl-alpha 0.0   TurboQuant rotation + low-bit quantization only
--qjl-alpha 0.25  conservative QJL residual correction
--qjl-alpha 1.0   full current correction estimate
```

This makes it possible to compare `lowbit`, `tq`, and `tq-qjl` under a similar
reconstruction-error tolerance and check which mode can use fewer bits.

The `lowbit` path uses GPU-side encoding and decoding:

```text
encode B_i on GPU
copy compressed payload to host
dequantize B_i_hat on GPU
reduce B += B_i_hat on GPU 0
```

This reduces the actual GPU-to-host payload for `--compress-b-bits 8` and
`--compress-b-bits 4`. The reconstructed block stays on device unless
`--no-check-b-error` is omitted.

The `tq` path is GPU-side:

```text
pad B_i to a power-of-two vector on GPU
apply random sign on GPU
apply normalized FWHT on GPU
low-bit encode/decode in the rotated domain on GPU
apply inverse normalized FWHT and inverse sign on GPU
```

The `tq-qjl` path is GPU-side for the expensive work: TQ reconstruction,
residual formation, QJL sign sketch, residual correction, and reduction into
GPU 0's `B`. The host receives compressed metadata, and it only receives
`B_i_hat` when local compression-error checking is enabled.

The program reports:

```text
reduce_B_quant_mode
reduce_B_quant_bits
reduce_B_qjl_dim
reduce_B_qjl_alpha
reduce_B_fp32_MiB
reduce_B_payload_MiB
reduce_B_compression_ratio
reduce_B_relative_error
reduce_B_compress_time_ms
```

Use `--no-check-b-error` for timing or profiler runs when the local
compression error is not needed. This skips the extra full-size `B_i`
device-to-host copy used only for:

```text
reduce_B_relative_error = ||B_i_hat - B_i||_F / ||B_i||_F
```

## CUDA Optimization Mode

For CUDA pipeline profiling, v2 also supports:

```text
--device-random-input
```

This generates each `A_i` block and the shared `Omega` matrix directly on the
owning GPU with a deterministic hash-based random generator. It avoids the
large host-side `A` allocation, row-block extraction, and `A_i` host-to-device
copy. Use it for profiling data movement and kernel scheduling, not for direct
numerical comparison with the host-random sweep because the random input
distribution is generated by a different code path.

Run the CUDA-focused sweep with:

```bash
sbatch run_randomized_svd_multigpu_v2_device_random.slurm
```

Profile the CUDA-focused path with Nsight Systems using:

```bash
DEVICE_RANDOM_INPUT=1 sbatch run_randomized_svd_multigpu_v2_nsys.slurm
```

Then inspect kernel-level behavior with Nsight Compute:

```bash
DEVICE_RANDOM_INPUT=1 sbatch run_randomized_svd_multigpu_v2_ncu.slurm
```

## Build

```bash
module load cuda/12.8
make
```

## Run

```bash
sbatch run_randomized_svd_multigpu_v2.slurm
```

The Slurm script runs each benchmark configuration with:

```text
none 0
lowbit 8
lowbit 4
lowbit 2
tq 8
tq 4
tq 2
tq-qjl 8, qjl_dim=64, qjl_alpha=0.01
tq-qjl 4, qjl_dim=64, qjl_alpha=0.01
tq-qjl 2, qjl_dim=64, qjl_alpha=0.01
```

This keeps the official runtime sweep on GPU-side compression.

The script is set to 2 GPUs because the current `contest_v100` QoS reports
`MaxTRESPU=gres/gpu=2`.

## Large 2-GPU Sweep

For bit/error curve experiments, use:

```bash
sbatch run_randomized_svd_multigpu_v2_large.slurm
```

This runs:

```text
m=32768, n=8192, k=256, oversample=64, l=320, ngpus=2
```

The large sweep now uses the fast Frobenius identity for end-to-end
reconstruction error, so it does not build `A_hat = U S V^T` on host.

The timing output includes both:

```text
subtotal_reported      full measured pipeline including host random matrix init
gpu_pipeline_reported  subtotal_reported minus init_host_random
```

Use `gpu_pipeline_reported` for CUDA and compression comparisons because host
random matrix generation can fluctuate across jobs.

It also reports compression-local error:

```text
reduce_B_relative_error = ||B_i_hat - B_i||_F / ||B_i||_F
```

against payload/bit rate. This is better for drawing the initial
bit-vs-error curve, while the fast end-to-end error checks the whole
randomized SVD result.

## Nsight Profiling

Use Nsight Systems first to see the full CPU/GPU timeline, cuBLAS/cuSOLVER
calls, CUDA memcpy, and kernel launch overhead:

```bash
sbatch run_randomized_svd_multigpu_v2_nsys.slurm
```

Use Nsight Compute after that to inspect kernel-level bottlenecks:

```bash
sbatch run_randomized_svd_multigpu_v2_ncu.slurm
```

The profiling scripts target both `tq 4-bit` and `tq-qjl 4-bit` by default.
They use `m=32768, n=8192, k=256, oversample=64, ngpus=2` to match the large
sweep, and pass `--no-check-b-error` so the profiler sees the timing path
without the extra error-only `B_i` copy. Use the generated `.nsys-rep` and
`.ncu-rep` files for offline analysis.

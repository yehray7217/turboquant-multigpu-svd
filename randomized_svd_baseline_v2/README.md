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

The current implementation uses host-mediated gathers/reductions for
portability. TurboQuant/QJL hooks are the `R_i` TSQR gather and the `B_i`
reduction.

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

## Build

```bash
module load cuda/12.8
make
```

## Run

```bash
sbatch run_randomized_svd_multigpu_v2.slurm
```

The script is set to 2 GPUs because the current `contest_v100` QoS reports
`MaxTRESPU=gres/gpu=2`.

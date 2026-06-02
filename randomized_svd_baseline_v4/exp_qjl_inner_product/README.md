# Experiment: Direct QJL/JL Inner Product Proxy

This is a small standalone experiment for testing whether QJL-style sketching is
promising inside

```text
B_i = Q_i^T A_i
```

before changing the main v3/v4 randomized SVD pipeline.

## Question

Current QJL is a postfix residual correction:

```text
B_i -> TQ(B_i) -> residual = B_i - decode(TQ(B_i)) -> QJL(residual)
```

This experiment tests a more direct inner-product formulation:

```text
B_i[r,c] = dot(Q_i[:,r], A_i[:,c])
```

with an implicit Rademacher sketch matrix `S`:

```text
SQ = S Q_i
SA = S A_i
B_hat = (SQ^T SA) / sketch_dim
```

If this direct sketch cannot approximate `Q_i^T A_i` well on small cases, it is
not worth integrating into the full distributed SVD pipeline yet.

## What It Measures

The program reports:

```text
relative_error = ||B_hat - B||_F / ||B||_F
exact_gemm_ms
sketch_Q_ms
sketch_A_ms
sketch_gemm_ms
total_direct_jl_ms
```

It also reports a `sign_proxy_error`, where `SQ` and `SA` are replaced by their
signs before the final GEMM. This is a rough 1-bit proxy, not a final QJL
implementation.

## Config

Default small case:

```text
m = 4096
n = 1024
l = 256
sketch_dim = 64, 128, 256, 512, 1024, 2048, 4096
```

Run:

```bash
sbatch sweep.slurm
```

Output files go to:

```text
output_logs/
error_logs/
```

## Interpretation

Useful signal:

```text
direct JL error << current TQ B error
```

Bad signal:

```text
direct JL error comparable to or worse than current TQ B error
```

The current TQ 4-bit B error in previous v2/v3 experiments is roughly `0.17` to
`0.18`. A direct inner-product sketch has to be competitive with that before it
is worth integrating into the main rSVD pipeline.

## Result: 2026-05-30

Run:

```text
job_id = 935018
GPU = 1 x Tesla V100-SXM2-32GB
m = 4096
n = 1024
l = 256
```

| sketch_dim | exact_gemm_ms | direct_jl_total_ms | direct_jl_relative_error | sign_proxy_relative_error |
|---:|---:|---:|---:|---:|
| 64 | 5.862 | 1.013 | 8.034 | 24.011 |
| 128 | 5.889 | 1.615 | 5.668 | 17.034 |
| 256 | 5.927 | 3.104 | 3.997 | 12.061 |
| 512 | 5.928 | 5.752 | 2.828 | 8.534 |
| 1024 | 5.931 | 11.206 | 1.996 | 6.073 |
| 2048 | 6.643 | 21.932 | 1.413 | 4.334 |
| 4096 | 5.873 | 43.048 | 0.998 | 3.128 |

Conclusion: this direct JL inner-product proxy is not competitive with the
current TQ 4-bit compression of `B_i`. The error remains much larger than
`0.17` to `0.18`, and making `sketch_dim` large enough to reduce error quickly
becomes slower than computing `B_i = Q_i^T A_i` directly.

This does not prove that every QJL variant is bad, but it does suggest that a
naive direct sketch of the `Q_i^T A_i` inner products should not be integrated
into the main v4 pipeline.

## Block-Wise QJL Result: 2026-05-30

Block-wise variant:

```text
split rows of Q_i and A_i into blocks
SQ_b = S_b Q_i[b]
SA_b = S_b A_i[b]
B_hat = sum_b SQ_b^T SA_b / sketch_dim
```

This improves GPU parallelism because each sketch output sums over a shorter
row block. It also allows the relative error to scale roughly with:

```text
sqrt(block_rows / sketch_dim)
```

Run:

```text
job_id = 935019, 935020
GPU = 1 x Tesla V100-SXM2-32GB
m = 4096
n = 1024
l = 256
```

Small sweep:

| block_rows | num_blocks | sketch_dim/block | total_sketch_dim | exact_gemm_ms | direct_jl_total_ms | direct_jl_relative_error |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 1 | 64 | 64 | 5.761 | 1.027 | 8.034 |
| 4096 | 1 | 128 | 128 | 6.664 | 1.618 | 5.668 |
| 4096 | 1 | 256 | 256 | 5.797 | 3.100 | 3.997 |
| 2048 | 2 | 64 | 128 | 5.880 | 0.880 | 5.675 |
| 2048 | 2 | 128 | 256 | 5.929 | 1.547 | 4.009 |
| 2048 | 2 | 256 | 512 | 6.622 | 2.955 | 2.822 |
| 1024 | 4 | 64 | 256 | 5.924 | 0.882 | 3.987 |
| 1024 | 4 | 128 | 512 | 6.633 | 1.554 | 2.815 |
| 1024 | 4 | 256 | 1024 | 5.966 | 2.937 | 1.992 |
| 512 | 8 | 64 | 512 | 6.621 | 0.846 | 2.824 |
| 512 | 8 | 128 | 1024 | 5.940 | 1.520 | 2.004 |
| 512 | 8 | 256 | 2048 | 5.957 | 2.866 | 1.414 |

Feasibility sweep near the TQ4 error range:

| block_rows | num_blocks | sketch_dim/block | total_sketch_dim | exact_gemm_ms | direct_jl_total_ms | direct_jl_relative_error |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 16 | 512 | 8192 | 5.951 | 5.917 | 0.703 |
| 256 | 16 | 1024 | 16384 | 5.840 | 11.451 | 0.497 |
| 128 | 32 | 1024 | 32768 | 5.910 | 12.291 | 0.352 |
| 128 | 32 | 2048 | 65536 | 6.571 | 24.961 | 0.249 |
| 64 | 64 | 2048 | 131072 | 6.596 | 28.504 | 0.177 |

Conclusion: block-wise QJL is much faster than naive dense JL at the same
effective error, but reaching TQ4-level error requires an enormous sketch. For
`block_rows=64` and `sketch_dim=2048`, the error reaches `0.1768`, but the
sketch dimension is `131072`, while the original `B_i` has only `l=256` rows.
That means the intermediate sketch representation is `512x` wider than `B_i`
before any quantization, and the runtime is about `4.3x` slower than exact
GEMM.

This makes block-wise QJL useful as a diagnostic, but not a good mainline
replacement for computing `B_i = Q_i^T A_i` exactly and compressing `B_i` with
TQ2/TQ4.

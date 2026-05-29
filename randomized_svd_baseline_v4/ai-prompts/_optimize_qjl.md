Please optimize and correct the `mode=tq-qjl` implementation.

Files to inspect and modify:

* `turboquant.cu`
* `turboquant.hpp`
* `randomized_svd_multigpu_v4.cu` if needed

Do NOT modify the timing metric redesign in this patch.

Do NOT modify the corrected reconstruction error metric in this patch.

Do NOT re-add Captured Energy Ratio.

Do NOT change `mode=tq`, except for refactoring shared helper functions if necessary.

Do NOT replace RHT in `mode=tq`; RHT should remain the preconditioning step for Lloyd-Max TQ.

# Goal

The current `mode=tq-qjl` is too slow and inaccurate. The likely cause is that QJL currently computes Gaussian coefficients inside CUDA kernels using hash-based random generation with expensive functions such as `logf`, `sqrtf`, and `cosf`.

This patch should optimize QJL by:

1. Removing per-entry hash Gaussian generation from hot kernels.
2. Pre-generating a fixed dense QJL matrix `S`.
3. Using cuBLAS GEMM for QJL projection and reconstruction.
4. Keeping residual norms and QJL scratch buffers on device.
5. Avoiding `cudaMalloc` / `cudaFree` inside hot encode/decode paths.
6. Preserving faithful QJL math.

# Background

Faithful TurboQuant-QJL should implement:

```
total b bits
=
(b - 1)-bit Lloyd-Max TQ-MSE
+
1-bit QJL residual sign sketch
```

Example:

```
tq-qjl 8-bit = 7-bit Lloyd-Max TQ-MSE + 1-bit QJL
tq-qjl 4-bit = 3-bit Lloyd-Max TQ-MSE + 1-bit QJL
```

QJL is designed for unbiased inner-product estimation, not necessarily better reconstruction MSE. Our current pipeline reconstructs the matrix before SVD, so QJL may still be a negative result. This patch is mainly to remove avoidable CUDA overhead and make the implementation faithful and measurable.

# Correct QJL math

For each vector:

```
x ∈ R^d
```

First run Lloyd-Max TQ-MSE with `mse_bits = total_bits - 1`:

```
x_mse = dequantize_tq_mse(quantize_tq_mse(x, mse_bits))
```

Then compute residual in the original vector space:

```
r = x - x_mse
```

Important:

* Residual must be computed after inverse RHT and norm rescaling.
* Do NOT compute residual only in the rotated domain unless equivalence is explicitly proven.
* The safe implementation is residual in original vector space.

Let:

```
S ∈ R^{qjl_dim x d}
```

be a fixed QJL Gaussian matrix.

Each row of `S` should be sampled from standard normal entries.

QJL encode:

```
projected = S r
q = sign(projected)
```

where:

```
q ∈ {-1, +1}^{qjl_dim}
```

Store:

* MSE Lloyd-Max indices using `mse_bits`
* original vector norm
* QJL sign bits
* residual norm `||r||_2`

QJL decode:

```
r_hat = sqrt(pi / 2) / qjl_dim * ||r||_2 * S^T q
```

If `qjl_alpha` exists, apply it as:

```
r_hat = qjl_alpha * sqrt(pi / 2) / qjl_dim * ||r||_2 * S^T q
```

Final reconstruction:

```
x_hat = x_mse + r_hat
```

# Important bit-rate note

If `qjl_dim == d`, then QJL signs use exactly 1 bit per coordinate.

If `qjl_dim != d`, the effective bit-rate changes:

```
effective_bits_per_coordinate
=
mse_bits + qjl_dim / d
```

This is okay for ablation, but the code must report or document this clearly.

Default should be:

```
qjl_dim = d
```

so that `tq-qjl b-bit` means approximately `b` bits per coordinate.

# Optimization requirement 1 — Pre-generate QJL matrix S

Do NOT generate Gaussian random numbers inside the projection/reconstruction kernels.

Instead:

1. Generate `S` once per supported `(d, qjl_dim, seed)` combination.
2. Store it on device.
3. Reuse it across vectors, blocks, repeats, encode, and decode.

Implementation options:

* Generate `S` on host deterministically, then copy to device once.
* Or generate `S` once on device in a setup kernel.
* Either is fine, but it must not happen inside the hot QJL encode/decode loop.

Do NOT load `S` from files.

Do NOT hard-code `S`.

Use deterministic seed-based generation so encoding and decoding use the same `S`.

# Optimization requirement 2 — Use cuBLAS GEMM

Current slow path likely computes QJL projection per vector or per coordinate.

Replace it with matrix operations.

Assume the compressed matrix is represented as many column vectors:

```
X ∈ R^{d x num_vectors}
```

After MSE dequantization, compute residual matrix:

```
R = X - X_mse
```

where:

```
R ∈ R^{d x num_vectors}
```

QJL projection should use GEMM:

```
P = S R
```

where:

```
S ∈ R^{qjl_dim x d}
R ∈ R^{d x num_vectors}
P ∈ R^{qjl_dim x num_vectors}
```

Use cuBLAS, preferably:

```
cublasSgemm
```

Then encode signs:

```
signs = sign(P)
```

During dequantization, unpack signs into a dense float matrix:

```
Q ∈ R^{qjl_dim x num_vectors}
```

where entries are `-1.0f` or `+1.0f`.

Then reconstruct residuals using GEMM:

```
R_hat = S^T Q
```

where:

```
S^T ∈ R^{d x qjl_dim}
Q   ∈ R^{qjl_dim x num_vectors}
R_hat ∈ R^{d x num_vectors}
```

Then scale each column:

```
R_hat[:, j] *= qjl_alpha * sqrt(pi / 2) / qjl_dim * residual_norm[j]
```

Finally:

```
X_hat = X_mse + R_hat
```

# Optimization requirement 3 — Device-side residual norms

QJL needs one residual norm per vector:

```
residual_norm[j] = ||R[:, j]||_2
```

Do not copy residual norms to host during hot path.

Store them as device array:

```
float* d_residual_norms
```

The payload for `mode=tq-qjl` must include these residual norms.

If the communication path requires host-side MPI payloads, copy the residual norm array as part of the payload in bulk, not one-by-one.

# Optimization requirement 4 — Avoid hot cudaMalloc/cudaFree

Do not allocate and free QJL scratch buffers inside every encode/decode call if avoidable.

Add reusable scratch buffers, for example:

* `d_residual`
* `d_projection`
* `d_signs_float`
* `d_reconstructed_residual`
* `d_residual_norms`
* packed QJL sign buffer

These buffers should be allocated once per block/worker/context and reused.

If introducing a scratch/context struct is too invasive, at least avoid repeated allocation inside inner loops.

# QJL payload layout

For `mode=tq-qjl`, payload must include:

1. Packed MSE Lloyd-Max indices using `mse_bits = bits - 1`
2. Original vector norms
3. Packed QJL sign bits
4. Residual norms

Payload byte accounting must include all four parts.

# Supported bits

For `mode=tq`:

```
bits = 2, 3, 4, 5, 6, 7, 8
```

For `mode=tq-qjl`:

```
total bits = 3, 4, 5, 6, 7, 8
```

because MSE uses `bits - 1`, and Lloyd-Max codebooks start at 2 bits.

If user requests:

```
mode=tq-qjl, bits=2
```

fail loudly before kernel launch.

# Supported dimensions

Supported codebook dimensions are:

```
d = 256, 512, 1024, 2048
```

Runtime vector dimension is usually:

```
d = l = k + oversample
```

For B compression:

```
B_i = Q_i^T A_i
shape = l x n
compress each column vector of length l
```

For Z compression:

```
Z_i = A_i^T Q_i
shape = n x l
compress each row vector of length l
```

If Z is stored as `n x l`, transpose or logically reinterpret it as `l x n` before applying column-vector TQ helpers.

If unsupported dimension is requested for `mode=tq` or `mode=tq-qjl`, fail loudly before kernel launch.

# Do not change these

* Do not replace RHT in TQ.
* Do not replace Lloyd-Max codebook TQ.
* Do not change corrected reconstruction error metric.
* Do not re-add Captured Energy Ratio.
* Do not modify timing metric redesign.
* Do not change raw / no-compression behavior.
* Do not change `mode=lowbit`.

# Optional but useful ablation support

If not already supported, preserve or add CLI/config support for:

```
--qjl-dim
--qjl-alpha
```

Recommended sweep later:

```
qjl_dim = 64, 128, 256
qjl_alpha = 0.25, 0.5, 1.0
```

But default should be:

```
qjl_dim = d
qjl_alpha = 1.0
```

If qjl_dim differs from d, make the effective bit-rate clear in logs.

# Diagnostics

Print a clear one-line diagnostic when `mode=tq-qjl` is active:

```
TQ-QJL active: dim=<d>, total_bits=<b>, mse_bits=<b-1>, qjl_dim=<qjl_dim>, qjl_alpha=<alpha>
```

Add checks if practical:

* MSE indices are within range.
* QJL signs unpack to `{-1, +1}`.
* residual norms are finite.
* dequantized output has no NaN / Inf.

# Acceptance criteria

After this patch:

1. `mode=tq` remains valid and unchanged except for shared helper refactoring.
2. `mode=tq-qjl` uses Lloyd-Max TQ-MSE with `bits - 1`.
3. `mode=tq-qjl` no longer uses max-abs uniform quantization.
4. QJL no longer generates Gaussian coefficients inside hot projection/reconstruction kernels.
5. QJL uses a pre-generated device matrix `S`.
6. QJL projection `S R` uses cuBLAS GEMM.
7. QJL reconstruction `S^T q` uses cuBLAS GEMM.
8. residual norms are per-vector and device-side.
9. payload byte accounting includes MSE indices, original norms, QJL sign bits, and residual norms.
10. `mode=tq-qjl --bits 2` fails loudly.
11. Code compiles cleanly.
12. The implementation is still allowed to show QJL as a negative result experimentally; the goal is correctness and removing obvious CUDA overhead, not forcing QJL to beat pure TQ.

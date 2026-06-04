Please implement faithful `mode=tq-qjl` support and update the generated Lloyd-Max codebook header.

Files to inspect and modify:

- `turboquant.cu`
- `turboquant.hpp`
- `tq_codebooks_generated.hpp`
- optionally add one script in the same `turboquant/` folder

Do NOT modify timing metrics in this patch.
Do NOT modify the corrected reconstruction error metric in this patch.
Do NOT re-add Captured Energy Ratio.
Do NOT replace RHT with dense random rotation.

# Context

The current `mode=tq` has already been migrated to:

    RHT preconditioning + Lloyd-Max scalar quantization + norm rescaling

The codebook arrays have already been extracted from `turboquant.cu` into a separate header:

    turboquant/tq_codebooks_generated.hpp

This header is included only by `turboquant.cu`.
Do not include it in `turboquant.hpp`.

The current `tq_codebooks_generated.hpp` already contains the `d = 256` Lloyd-Max codebook arrays.

The codebook set has now been expanded. We now have precomputed JSON codebooks for:

    dimensions = 256, 512, 1024, 2048
    bits       = 2, 3, 4, 5, 6, 7, 8

The JSON files are under:

    turboquant/codebook/

with names like:

    codebook_d256_b2.json
    codebook_d512_b8.json
    codebook_d2048_b7.json

Each JSON has the official TurboQuant reference format:

    centroids
    boundaries
    mse_per_coord
    mse_total
    d
    bits

Do NOT load JSON files at runtime.

# Task 1 — Update tq_codebooks_generated.hpp

Please add the missing codebooks for:

    d = 512, 1024, 2048
    bits = 2, 3, 4, 5, 6, 7, 8

The existing `d = 256` arrays are already present.

To avoid wasting tokens and avoid manual copy-paste, write a small script in the same folder, for example:

    turboquant/generate_tq_codebooks_header.py

The script should read JSON files from:

    turboquant/codebook/

and update or regenerate:

    turboquant/tq_codebooks_generated.hpp

It is acceptable to regenerate the whole header from JSON files, as long as:

- `#pragma once` is preserved.
- the file remains a generated C++ header.
- all supported `(dim, bits)` pairs are included.
- there is no runtime JSON loading.
- the generated arrays are committed into the header.

The generated arrays should remain device-side constants, for example:

    __device__ __constant__ float d256_b4_centroids[16] = { ... };
    __device__ __constant__ float d256_b4_boundaries[17] = { ... };

Also add a device-side helper using `switch case`, for example:

    struct TQDeviceCodebook {
        const float* centroids;
        const float* boundaries;
        int levels;
    };

    __device__ __forceinline__
    TQDeviceCodebook get_tq_codebook_device(int dim, int bits);

The helper should support:

    dim  = 256, 512, 1024, 2048
    bits = 2, 3, 4, 5, 6, 7, 8

Return `{nullptr, nullptr, 0}` for unsupported combinations.

However, do not rely on device-side errors alone.

Before launching any TQ / TQ-QJL kernel, perform host-side validation:

    dim must be one of 256, 512, 1024, 2048
    mode=tq bits must be 2..8
    mode=tq-qjl total bits must be 3..8

Fail loudly on unsupported configurations before kernel launch.

# Task 2 — Keep mode=tq working

Make sure existing `mode=tq` still works after moving to the generated codebook header.

For each vector `x ∈ R^d`, `mode=tq` should do:

1. Compute and store original L2 norm:

       norm = ||x||_2

2. Normalize:

       u = x / (norm + eps)

3. Apply existing RHT:

       y = RHT(u)

4. For each coordinate `y_j`, find bucket index `idx_j`:

       boundaries[idx_j] <= y_j < boundaries[idx_j + 1]

5. Store / bit-pack `idx_j`.

Dequantization:

1. Unpack `idx_j`.
2. Reconstruct rotated-domain coordinate:

       y_hat_j = centroids[idx_j]

3. Apply inverse RHT:

       u_hat = inverse_RHT(y_hat)

4. Rescale:

       x_hat = norm * u_hat

Important bucket rule:

    centroids length = 2^bits
    boundaries length = 2^bits + 1

Bucket `i` is:

    [boundaries[i], boundaries[i + 1])

and maps to:

    centroids[i]

The last bucket should include the right endpoint.

Do not use the old max-abs uniform quantization path inside `mode=tq`.

The old formula:

    scale = max(abs(x)) / qmax
    q = round(x / scale)
    x_hat = q * scale

may remain only for `mode=lowbit`.

# Task 3 — Implement faithful mode=tq-qjl

Implement faithful TurboQuant-prod / QJL behavior.

For total bit-width `b`, `tq-qjl` means:

    (b - 1)-bit Lloyd-Max TQ-MSE
    +
    1-bit QJL residual sign sketch

Examples:

    tq-qjl 8-bit = 7-bit Lloyd-Max TQ-MSE + 1-bit QJL
    tq-qjl 4-bit = 3-bit Lloyd-Max TQ-MSE + 1-bit QJL
    tq-qjl 3-bit = 2-bit Lloyd-Max TQ-MSE + 1-bit QJL

Since the available Lloyd-Max codebooks start at 2 bits:

    tq-qjl --bits 2 is unsupported

If the user requests `mode=tq-qjl` with `bits=2`, fail loudly before kernel launch.

Supported `mode=tq-qjl` total bits:

    bits = 3, 4, 5, 6, 7, 8

# Faithful QJL algorithm

For each vector:

    x ∈ R^d

## Encoding

1. Run Lloyd-Max TQ-MSE using `b - 1` bits.

This should reuse the same TQ-MSE path as `mode=tq`, except with `mse_bits = b - 1`.

The temporary MSE reconstruction is:

    x_mse = dequantize_tq_mse(quantize_tq_mse(x, b - 1))

2. Compute residual in original vector space:

    r = x - x_mse

Important:

- The residual must be computed after inverse RHT and norm rescaling.
- Do not compute residual only in the rotated domain unless you prove equivalence.
- The simplest correct implementation is residual in original vector space.

3. Compute and store residual norm:

    residual_norm = ||r||_2

4. Generate / use a fixed shared QJL matrix:

    S ∈ R^{d x d}

Use deterministic generation from:

    dimension d
    fixed seed
    device

Do NOT hard-code the QJL matrix.
Do NOT load the QJL matrix from files.
Do NOT use runtime file I/O.

The same S must be used consistently for encoding and decoding.

For the first faithful implementation, a dense standard Gaussian QJL matrix is acceptable.

5. Compute QJL signs:

    q = sign(S r)

where:

    q_j ∈ {-1, +1}

6. Pack `q` as 1 bit per coordinate.

The `tq-qjl` payload must store:

- MSE Lloyd-Max indices using `b - 1` bits per coordinate
- original vector norms
- QJL sign bits using 1 bit per coordinate
- residual norms

## Dequantization

1. Reconstruct MSE approximation:

    x_mse = dequantize_tq_mse(mse_indices, norm, b - 1)

2. Unpack QJL signs:

    q ∈ {-1, +1}^d

3. Reconstruct residual estimate:

    r_hat = sqrt(pi / 2) / d * residual_norm * S^T q

4. Return:

    x_hat = x_mse + r_hat

# Important notes about QJL

QJL is designed to make inner-product estimation unbiased.

It is not guaranteed to improve reconstruction MSE.

Therefore, do not assume `tq-qjl` must produce lower reconstruction error than `tq`.

This project currently uses a dequantize-to-matrix communication pipeline, so implement the reconstruction formula above first.

A later improvement may integrate QJL directly into inner-product estimation, but that is not required in this patch.

# Payload byte accounting

Update payload byte accounting.

For `mode=tq`, payload includes:

- packed Lloyd-Max indices
- original vector norms

For `mode=tq-qjl`, payload includes:

- packed MSE Lloyd-Max indices using `b - 1` bits
- original vector norms
- packed QJL sign bits using 1 bit per coordinate
- residual norms

Make sure compression ratio calculations include all of these.

# Supported dimensions

The runtime vector dimension is usually:

    d = l = k + oversample

For B compression:

    B_i = Q_i^T A_i
    shape = l x n
    compress each column vector of length l

For Z compression:

    Z_i = A_i^T Q_i
    shape = n x l
    compress each row vector of length l

If needed, transpose Z to l x n internally so the existing column-vector TQ helper can be reused.

Supported dimensions are:

    d = 256, 512, 1024, 2048

If `mode=tq` or `mode=tq-qjl` is requested for an unsupported vector dimension, fail loudly before kernel launch.

# Implementation suggestions

1. Refactor common TQ-MSE encode/decode helpers so both `tq` and `tq-qjl` can reuse them.
2. Use `get_tq_codebook_device(dim, bits)` inside kernels.
3. Validate `dim` and `bits` on the host before launching kernels.
4. Use `switch case` in the device-side codebook helper.
5. Keep RHT as the rotation/preconditioning step.
6. Do not include `tq_codebooks_generated.hpp` from `turboquant.hpp`.
7. Include `tq_codebooks_generated.hpp` only from `turboquant.cu`.

# Diagnostics

Add simple diagnostics if easy:

- Check indices are in `[0, 2^bits - 1]`.
- Check QJL sign bits unpack to `{-1, +1}`.
- Check dequantized output has no NaN / Inf.
- Print a clear line when `mode=tq-qjl` is active, including:
  - total bits
  - MSE bits
  - vector dimension
  - codebook used

# Acceptance criteria

After this patch:

1. `mode=tq` still works with Lloyd-Max codebooks.
2. `mode=tq-qjl` no longer uses the old uniform max-abs path.
3. `mode=tq-qjl --bits 8` uses 7-bit Lloyd-Max MSE + 1-bit QJL residual.
4. `mode=tq-qjl --bits 4` uses 3-bit Lloyd-Max MSE + 1-bit QJL residual.
5. `mode=tq-qjl --bits 2` fails loudly.
6. Codebooks for dimensions 256, 512, 1024, 2048 and bits 2 through 8 are hard-coded through generated C++ arrays.
7. No runtime JSON file loading is introduced.
8. Payload byte accounting includes MSE indices, original norms, QJL signs, and residual norms.
9. The code compiles cleanly.
10. Existing `mode=tq` results remain valid.
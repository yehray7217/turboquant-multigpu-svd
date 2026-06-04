Please modify `turboquant.cu` so that `mode=tq` uses the correct TurboQuant MSE quantization flow with precomputed Lloyd-Max codebooks.

Important context:
- Keep the current rotation implementation as RHT / randomized Hadamard transform. Do NOT replace it with dense random rotation.
- The current RHT implementation has already been validated and should remain the rotation/preconditioning step.
- I have already generated Lloyd-Max codebooks under `turboquant/codebook/`.
- The codebook files are named like:
  - `codebook_d256_b2.json`
  - `codebook_d256_b3.json`
  - ...
  - `codebook_d256_b8.json`
- The JSON format matches the official TurboQuant reference implementation:
  - `centroids`: Lloyd-Max reconstruction values
  - `boundaries`: decision boundaries
  - `d`: vector dimension
  - `bits`: bit-width
- Do NOT load these JSON files at runtime.
- Instead, hard-code the centroids and boundaries directly into `turboquant.cu` as static arrays.

Main goal:
- Replace the current TQ max-abs uniform quantization with Lloyd-Max codebook quantization.
- Keep `lowbit` mode unchanged.
- Only `mode=tq` should use Lloyd-Max codebooks.

Current compression targets:
1. `B_i = Q_i^T A_i`
   - Shape is logically `l x n`.
   - Compress each column vector of length `l`.
   - Current main setting: `l = 256`, so use `codebook_d256_b{bits}`.

2. Subspace iteration `Z_i = A_i^T Q_i`
   - Shape is logically `n x l`.
   - I have already requested that Z compression should compress row vectors.
   - Therefore each compressed Z row vector also has length `l = 256`.
   - So Z compression should also use `codebook_d256_b{bits}`.

Required quantization flow for each vector `x ∈ R^256`:
1. Compute and store its L2 norm:
   `norm = ||x||_2`
2. Normalize:
   `u = x / (norm + eps)`
3. Apply the existing RHT:
   `y = RHT(u)`
4. For each coordinate `y_j`, find index `idx_j` such that:
   `boundaries[idx_j] <= y_j < boundaries[idx_j + 1]`
5. Store / bit-pack `idx_j`.

Required dequantization flow:
1. Unpack `idx_j`.
2. Reconstruct rotated-domain coordinate:
   `y_hat_j = centroids[idx_j]`
3. Apply inverse RHT:
   `u_hat = inverse_RHT(y_hat)`
4. Rescale:
   `x_hat = norm * u_hat`

Important bucket rule:
- `centroids` has length `2^bits`.
- `boundaries` has length `2^bits + 1`.
- Bucket `i` is:
  `[boundaries[i], boundaries[i+1])`
- Bucket `i` maps to:
  `centroids[i]`
- The last bucket should include the right endpoint:
  if `y_j >= boundaries[last - 1]`, map to the last centroid.

Supported dimensions and bits:
- For now, only support vector dimension `d = 256`.
- If `mode=tq` is requested for a vector dimension other than 256, fail loudly with an error.
- Support at least:
  `bits = 2, 3, 4, 5, 6, 7, 8`
- If unsupported bits are requested, fail loudly with an error.

Remove / bypass old TQ formula:
- The current formula:
  `scale = max(abs(x)) / qmax`
  `q = round(x / scale)`
  `x_hat = q * scale`
  is uniform max-abs quantization and is NOT Lloyd-Max TurboQuant.
- This old formula may remain for `lowbit`, but must not be used for `mode=tq`.

Hard-code implementation requirements:
- Add static arrays in `turboquant.cu` for:
  - `d256_b2_centroids`
  - `d256_b2_boundaries`
  - ...
  - `d256_b8_centroids`
  - `d256_b8_boundaries`
- Add a helper function that selects the correct codebook by `bits`.
- A simple linear scan over boundaries is acceptable first.
- Binary search is optional.

Diagnostics:
- Add or preserve a small diagnostic to verify:
  - all indices are within `[0, 2^bits - 1]`
  - quantize/dequantize does not produce NaN
  - reconstruction MSE is finite
- If possible, print whether Lloyd-Max TQ path is active, so we can distinguish it from the old uniform path in logs.

Comments / documentation:
- Update comments so `mode=tq` is described as:
  `RHT preconditioning + Lloyd-Max scalar quantization + norm rescaling`
- Do not describe `mode=tq` as simple uniform low-bit quantization anymore.
- Mention that the codebooks are precomputed Lloyd-Max optimal scalar quantizers for the coordinate distribution after random rotation / RHT of a unit vector.
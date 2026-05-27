# AI Takeaway — turboquant/

## Purpose

**Shared compression kernel library.** All three SVD implementations (v1/v2/v3)
call into this directory's CUDA kernels for TurboQuant and QJL compression/
decompression of `B_i` blocks.

This is the place to modify if you want to:
- Add new quantization strategies
- Improve FWHT performance
- Fix the QJL residual estimator
- Add per-column or per-row scaling

---

## Files

| File | Description |
|------|-------------|
| `turboquant.hpp` | Public C++ API header. All types and function signatures. |
| `turboquant.cu` | ~1842 lines. All CUDA kernels + host-side helpers. |
| `README.md` | Usage examples, supported modes, planned extensions. |

---

## Key Concepts

### Compression Modes

| Mode | Bits | What Happens |
|------|------|-------------|
| `kNone` | 0 | Passthrough, FP32 |
| `kLowBit` | 8/4/2 | Per-block symmetric scalar quantization |
| `kTurboQuant` | 8/4/2 | Random signs + FWHT + low-bit quant + inverse |
| `kTurboQuantQjl` | 8/4/2 | TQ + QJL 1-bit residual sketch correction |

### Low-Bit Quantization Formula

```
qmax = 127 (8-bit), 7 (4-bit), 3 (2-bit)
scale = max(abs(x)) / qmax
q = clamp(round(x / scale), -qmax, qmax)
x_hat = q * scale
```
One global scale per B_i block (per-block, not per-column or per-element).

### TurboQuant (TQ) Pipeline

```
Input x (l-dim vector per column of B_i)
  1. Apply Rademacher random signs: x' = s ⊙ x   (signs deterministic from seed)
  2. Pad x' to next power of 2
  3. Normalized FWHT: x'' = H x' / sqrt(d)
  4. Low-bit quantize x'' → codes
  5. [decode side] inverse FWHT + inverse signs → x_hat
```

FWHT mixes all coordinates, making the distribution more uniform and reducing
the max-abs/RMS ratio — this improves quantization efficiency vs raw lowbit.

### TQ+QJL (Exploratory)

```
x_hat_TQ = TQ_decode(TQ_encode(x))
residual r = x - x_hat_TQ
G = Gaussian random matrix (seed-deterministic)
q = sign(G r)           [1-bit sketch of residual]
r_hat ≈ (sqrt(π/2)/d_qjl) * Gᵀ q   [approximate reconstruction]
x_hat = x_hat_TQ + alpha * r_hat
```

**Current status**: does not improve B error. Any nonzero `alpha` worsens results.
The residual reconstruction `r_hat` is too crude — it accumulates sign errors.

### Fused Decode-Add Kernel (Key Optimization)

When `padded_rows <= 1024`, the decode-add path uses a **shared-memory fused
kernel** (`dequantize_column_tq_payload_add_to_fp32`):
- Loads packed int codes into shared memory
- Dequantizes in shared memory
- Runs inverse FWHT in shared memory
- Applies inverse signs
- Atomically accumulates into `d_accumulator` (B on GPU 0)

This removes one global memory round-trip vs the unfused path.
Benefit: −0.68 ms (4-bit) / −1.47 ms (2-bit) on 16-GPU large run.

---

## Public API Summary

```cpp
#include "turboquant/turboquant.hpp"
using namespace turboquant;

// Build options
QuantizeOptions opts = make_quantize_options(bits, "tq", 0, 0.0f, seed);

// Host-side round-trip (testing/correctness)
CompressedBlock block = quantize_fp32_block(values, rows, cols, opts);
std::vector<float> recon = dequantize_fp32_block(block);

// GPU-side encode only (returns host-side CompressedBlock with codes on host)
CompressedBlock block = quantize_fp32_device_block(d_values, rows, cols, opts, stream);

// GPU-side encode + decode to device buffer (used by v2/v3 pipeline)
CompressedBlock block = quantize_dequant_fp32_device_block_to_device(
    d_values, rows, cols, opts, d_reconstructed, stream);

// GPU-side encode to device payload (used by v3 for compressed MPI)
DeviceCompressedBlock dblk = quantize_fp32_device_column_tq_to_device_payload(
    d_values, rows, cols, opts, d_codes, d_work, d_signs, nullptr, stream);

// GPU-side decode + accumulate (fused kernel, used by v3 on rank 0)
dequantize_column_tq_payload_add_to_fp32(dblk, d_accumulator, d_work, d_signs, stream);
```

### Key Types

```cpp
struct CompressedBlock {
    int rows, cols, bits, qjl_dim, padded_count;
    float scale, residual_norm, qjl_alpha;
    unsigned seed;
    QuantizeMode mode;
    std::vector<uint8_t> codes;      // packed integer codes (host)
    std::vector<uint8_t> qjl_signs; // 1-bit QJL sketch (host)
};

struct DeviceCompressedBlock {
    // Same scalars as CompressedBlock, but codes/signs live on device:
    uint8_t* d_codes;
    int* d_qjl_signs;
};
```

---

## Important Internal Functions

| Function | What It Does |
|----------|-------------|
| `qmax_for_bits(b)` | Returns 127, 7, 3 for 8/4/2-bit |
| `next_power_of_two(n)` | Padding for FWHT |
| `initialize_column_tq_signs(rows, cols, seed, d_signs)` | Pre-generate per-column Rademacher signs |
| `copy_pad_random_sign_kernel` | Apply signs + pad to power-of-2 |
| `quantize_int8/int4/int2_pack_kernel` | Pack codes into bytes |
| `fwht_column_kernel` | Normalized Walsh-Hadamard on columns |
| `dequantize_column_tq_fused_decode_add_kernel` | Fused decode+accumulate (key) |

---

## Current Status

- `lowbit`: works, stable
- `tq`: works, optimized (fused decode-add, removed tail sync)
- `tq-qjl`: works mechanically but **does not improve error** — treat as negative result

---

## Next Steps / What To Try

1. **Per-column scaling**: replace single per-block `scale` with one scale per column
   of B_i. Reduces quantization error at the same bit width (especially at TQ 2-bit).
2. **GPU-side FWHT on larger vectors**: current fused kernel works for `padded_rows <= 1024`.
   For larger l, it falls back to global memory. A multi-block FWHT kernel could
   extend the fast path.
3. **QJL redesign**: instead of correcting the reconstructed B_i after TQ, integrate
   QJL into the inner-product computation for `B_i = Qᵢᵀ Aᵢ` directly. This is
   the formulation in the original TurboQuant/QJL paper.
4. **FP16 quantize/dequantize**: add a half-precision lowbit path to compare TQ
   against the simpler 16-bit baseline.
5. **Profile with Nsight Compute**: run `randomized_svd_baseline_v2/run_randomized_svd_multigpu_v2_ncu.slurm`
   and inspect occupancy, memory throughput, and warp efficiency of the FWHT and
   pack/unpack kernels.

# CUDA Optimization Round 1

This experiment tracks the first CUDA optimization round after v4 switched to
RHT + Lloyd-Max TurboQuant.

For lower-noise TQ-only kernel timing, use
`../exp_tq_kernel_microbench`. That microbenchmark excludes QJL and the full
RSVD pipeline, and measures only TQ encode/decode roundtrip timing.

## Policy

Round 1 may include both TQ-specific CUDA kernels and shared pipeline kernels.
However, every shared-pipeline optimization must be checked against both
`none` and `TQ`. If it improves `none` much more than `TQ` and weakens the
relative TQ speedup story, keep it out of round 1 and revisit it in a later
shared-baseline optimization round.

## Change: TQ4/TQ2 Specialized Bitpack Kernels

The previous TQ Lloyd-Max payload path used the generic bit writer:

```text
bitpack_write_code(...)
```

That writer supports arbitrary bit widths, but it uses `atomicOr` into packed
32-bit words. This is flexible, but it is unnecessary for the common TQ4 and
TQ2 paths.

Round 1 adds specialized CUDA kernels:

```text
column_tq_lloyd_quantize_pack4_kernel
column_tq_lloyd_quantize_pack2_kernel
column_tq_lloyd_dequantize_pack4_kernel
column_tq_lloyd_dequantize_pack2_kernel
```

For TQ4, each thread packs two 4-bit Lloyd-Max codes into one byte. For TQ2,
each thread packs four 2-bit Lloyd-Max codes into one byte. This removes the
generic atomic bit writer for these modes and avoids clearing `d_codes` before
quantization.

TQ3 and other non-byte-aligned modes still use the generic bitpack path.

## Validation Plan

Use:

```bash
sbatch run_pack4_eval_8gpu.slurm
```

The script runs:

```text
none
TQ4
```

with the same v4 8-GPU config used in the TQ4 vs INT4 comparison:

```text
m = 65536
n = 16384
k = 250
oversample = 6
l = 256
ngpus = 8
subspace_iter = 1
repeat = 30
```

Compare against the pre-change reference from job `935021`:

| Method | Total Time | B Error | Final Error | Error Ratio |
|---|---:|---:|---:|---:|
| none | 192.389 ms | skipped | 42.9953% | 1.02922 |
| TQ4 | 129.813 ms | 13.1359% | 44.6350% | 1.06847 |

Expected outcome: TQ4 should improve or stay neutral, while `none` should be
unchanged within run-to-run noise. B error and final reconstruction error should
remain unchanged.

## Result: 2026-06-02

Run:

```text
job_id = 942626
node = gn1216
GPU = 8 x V100
```

| Method | Total Time | GPU Compute | Host/Staging | NVLink | IB | B Error | Final Error | Error Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 197.341 ms | 99.5711 ms | 90.1014 ms | 0.691394 ms | 6.6437 ms | skipped | 42.9953% | 1.02922 |
| TQ4 | 142.128 ms | 93.3694 ms | 46.2815 ms | 0.472972 ms | 1.46875 ms | 13.1359% | 44.6350% | 1.06847 |

Compared with the pre-change reference:

| Method | Pre-change | Pack-specialized trial | Change |
|---|---:|---:|---:|
| none | 192.389 ms | 197.341 ms | +4.952 ms |
| TQ4 | 129.813 ms | 142.128 ms | +12.315 ms |

Accuracy stayed unchanged, but the TQ4 timing got worse. The likely cause is
that the specialized pack kernels reduced thread-level parallelism and made
each thread do multiple Lloyd-Max bucket searches, while the original generic
bit writer kept one thread per coordinate. Avoiding the `atomicOr` bit writer
did not compensate for the reduced parallelism.

Decision: **fallback**. The code change was reverted; this experiment is kept
only as a negative result.

## Change: Fused Shared-Memory FWHT for `rows == 256`

The v4 TQ path uses column-wise randomized Hadamard transforms. Before this
change, each transform was executed as many small CUDA kernels:

```text
normalize/sign kernel
8 FWHT stage kernels
scale kernel
```

Decode had the same issue:

```text
centroid decode kernel
8 FWHT stage kernels
scale/sign/norm add or store kernel
```

For the current RSVD runs, the TQ vector dimension is `l = 256`. This trial
adds a specialized shared-memory path for `rows == 256`:

```text
column_tq_normalize_sign_fwht256_kernel
column_tq_lloyd_dequantize_fwht256_apply_add_kernel
column_tq_lloyd_dequantize_fwht256_apply_store_kernel
```

Each CUDA block owns one column vector, loads 256 values into shared memory,
runs all FWHT butterfly stages inside the block, applies the normalized scale,
and writes the final result. Other dimensions still use the original generic
FWHT path.

This is a TQ/TQ+QJL-only optimization. It does not change `none`.

## Result: 2026-06-02

Run:

```text
job_id = 942656
node = gn1220
GPU = 8 x V100
```

| Method | Total Time | GPU Compute | Host/Staging | NVLink | IB | B Error | Final Error | Error Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 191.555 ms | 95.8829 ms | 88.1774 ms | 0.672009 ms | 6.5627 ms | skipped | 42.9953% | 1.02922 |
| TQ4 | 119.115 ms | 79.5231 ms | 36.8331 ms | 0.488218 ms | 1.4118 ms | 13.1359% | 44.6350% | 1.06847 |

Compared with the pre-change reference:

| Method | Pre-change | Fused-FWHT trial | Change |
|---|---:|---:|---:|
| none | 192.389 ms | 191.555 ms | -0.834 ms |
| TQ4 | 129.813 ms | 119.115 ms | -10.698 ms |

Accuracy stayed unchanged. Unlike the previous trials, this one improves the
TQ path directly while leaving `none` essentially unchanged. The improvement
comes from removing many small FWHT stage launches and avoiding repeated global
memory passes through the 256-element TQ vectors.

Decision: **keep** for round 1. This should also benefit TQ+QJL because its MSE
reconstruction path uses the same TQ inverse transform.

## TQ+QJL4 Check After Fused FWHT

Run:

```text
job_id = 942657
node = gn1220
GPU = 8 x V100
```

Configuration:

```text
m = 65536
n = 16384
k = 250
oversample = 6
l = 256
ngpus = 8
subspace_iter = 1
compress_b_mode = tq-qjl
compress_b_bits = 4
compress_subspace_mode = tq-qjl
compress_subspace_bits = 4
qjl_dim = 256
qjl_alpha = 1.0
repeat = 30
```

| Method | Total Time | GPU Compute | Host/Staging | NVLink | IB | B Error | Final Error | Error Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TQ+QJL4 | 144.062 ms | 96.3977 ms | 45.2128 ms | 0.592939 ms | 1.34942 ms | 33.7048% | 52.8646% | 1.26547 |

The run completed successfully with stable timing and no accuracy regression
from the fused FWHT path. TQ+QJL4 is still slower and less accurate than TQ4 in
this RSVD workload:

| Method | Total Time | B Error | Final Error | Error Ratio |
|---|---:|---:|---:|---:|
| TQ4 | 119.115 ms | 13.1359% | 44.6350% | 1.06847 |
| TQ+QJL4 | 144.062 ms | 33.7048% | 52.8646% | 1.26547 |

This supports the current conclusion: fused FWHT is worth keeping, but QJL4
itself remains a weaker tradeoff than TQ4 for the current `B_i` / subspace
compression workload.

## Change: Fuse FWHT Encode and Quantize

After the successful fused-FWHT path, this trial tried to go one step further:
for `rows == 256`, directly run:

```text
normalize/sign
FWHT
Lloyd-Max bucket lookup
bitpack write
```

inside one shared-memory kernel. The goal was to remove the intermediate
`d_work` write/read between the fused FWHT kernel and the Lloyd-Max quantize
kernel.

## Result: 2026-06-02

Run:

```text
job_id = 942659
node = gn1220
GPU = 8 x V100
```

| Method | Total Time | GPU Compute | Host/Staging | NVLink | IB | B Error | Final Error | Error Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 190.771 ms | 96.8569 ms | 86.2872 ms | 0.668727 ms | 6.70815 ms | skipped | 42.9953% | 1.02922 |
| TQ4 | 128.249 ms | 79.2742 ms | 46.2137 ms | 0.432419 ms | 1.45587 ms | 13.1359% | 44.6350% | 1.06847 |

Compared with the kept fused-FWHT result:

| Method | Fused FWHT | Fused FWHT + quantize | Change |
|---|---:|---:|---:|
| none | 191.555 ms | 190.771 ms | -0.784 ms |
| TQ4 | 119.115 ms | 128.249 ms | +9.134 ms |

Accuracy stayed unchanged, but TQ4 got slower. The likely cause is that mixing
the full-column shared-memory FWHT block with global atomic bitpack writes
creates a less favorable kernel shape. Keeping Lloyd-Max quantization as the
separate one-thread-per-coordinate kernel is faster for this workload.

Decision: **fallback to fused FWHT only**. The direct encode+quantize fusion was
reverted.

## Change: Fuse Norm Reduction into FWHT Encode

The kept fused-FWHT path still computes column norms in a separate kernel before
the shared-memory FWHT kernel. This trial moved the `rows == 256` norm reduction
into the same per-column block that performs normalize/sign/FWHT:

```text
norm reduction
normalize/sign
FWHT
write transformed d_work
```

The goal was to remove one TQ encode kernel launch while still keeping Lloyd-Max
quantization separate.

## Result: 2026-06-02

Run:

```text
job_id = 942676
node = gn1215
GPU = 8 x V100
```

| Method | Total Time | GPU Compute | Host/Staging | NVLink | IB | B Error | Final Error | Error Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 200.610 ms | 101.762 ms | 90.9339 ms | 0.690369 ms | 6.84935 ms | skipped | 42.9953% | 1.02922 |
| TQ4 | 129.115 ms | 81.6642 ms | 44.8307 ms | 0.437208 ms | 1.41562 ms | 13.1359% | 44.6350% | 1.06847 |

Compared with the kept fused-FWHT result:

| Method | Fused FWHT | Fused norm + FWHT | Change |
|---|---:|---:|---:|
| none | 191.555 ms | 200.610 ms | +9.055 ms |
| TQ4 | 119.115 ms | 129.115 ms | +10.000 ms |

The node for this run was slower overall, but TQ4 was still clearly worse than
the kept fused-FWHT result. Folding the norm reduction into the transform block
made the kernel heavier and raised GPU compute time. The saved launch did not
pay for the extra reduction work inside the fused kernel.

Decision: **fallback to fused FWHT only**. The norm-fusion change was reverted.

## Change: Warp-Shuffle FWHT Inside the Fused 256-D Kernel

The kept fused-FWHT implementation still used shared memory for all 8 Hadamard
butterfly stages. For a 256-dimensional vector, the first 5 stages are entirely
inside each 32-thread warp. This trial changes the fused FWHT implementation:

```text
len = 1, 2, 4, 8, 16     -> warp shuffle
len = 32, 64, 128        -> shared memory + block sync
```

The goal is to reduce shared-memory traffic and reduce `__syncthreads()` calls
inside the TQ/TQ+QJL fused transform kernels. This only changes the internal
CUDA implementation of the fused FWHT; it should not change payload size or
accuracy.

Validation command:

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4
module load cuda/12.8
module load ucx/1.14.1
module load openmpi/5.0.2_ucx1.14.1_cuda12.3
make
cd exp_cuda_opt_round1
sbatch run_pack4_eval_8gpu.slurm
```

## Result: 2026-06-02

Run:

```text
job_id = 942694
node = gn1216
GPU = 8 x V100
```

| Method | Total Time | GPU Compute | Host/Staging | NVLink | IB | B Error | Final Error | Error Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 196.830 ms | 97.5791 ms | 91.3969 ms | 0.672301 ms | 6.83463 ms | skipped | 42.9953% | 1.02922 |
| TQ4 | 124.997 ms | 76.6643 ms | 45.9390 ms | 0.441920 ms | 1.41481 ms | 13.1359% | 44.6350% | 1.06847 |

Compared with the kept fused-FWHT result:

| Method | Fused FWHT | Warp-shuffle FWHT | Change |
|---|---:|---:|---:|
| none | 191.555 ms | 196.830 ms | +5.275 ms |
| TQ4 | 119.115 ms | 124.997 ms | +5.882 ms |

Accuracy stayed unchanged. The warp-shuffle version reduced the reported GPU
compute time for TQ4, but total runtime was still worse than the kept
shared-memory fused-FWHT implementation. Since the end-to-end TQ4 timing did
not improve, this does not satisfy the round-1 keep rule.

Decision: **fallback to shared-memory fused FWHT**. The warp-shuffle change was
reverted.

## Change: Branchless TQ4 Lloyd-Max Bucket Lookup

Nsight Compute showed that `column_tq_lloyd_quantize_kernel` was not saturating
memory bandwidth and had low not-predicated threads per warp. The isolated TQ
microbenchmark then tested a branchless TQ4 bucket lookup that keeps the
existing one-thread-per-coordinate bitpack path.

Microbenchmark result:

| rows | cols | quantize_bitpack | quantize_branchless4_alt | Speedup |
|---:|---:|---:|---:|---:|
| 256 | 16384 | 0.0989 ms | 0.0689 ms | 30.3% |
| 256 | 32768 | 0.1680 ms | 0.1196 ms | 28.8% |

This change moves the branchless lookup into the main quantize kernel when
`bits == 4`. Other bit widths still use the original lookup.

Validation command:

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4
module load cuda/12.8
module load ucx/1.14.1
module load openmpi/5.0.2_ucx1.14.1_cuda12.3
make
cd exp_cuda_opt_round1
sbatch run_pack4_eval_8gpu.slurm
```

Decision: **pending full RSVD validation**. Keep if TQ4 improves over the kept
fused-FWHT reference (`119.115 ms`) and B/final errors remain unchanged.

## Result: 2026-06-02

Run:

```text
job_id = 942938
node = gn1215
GPU = 8 x V100
```

| Method | Total Time | GPU Compute | Host/Staging | NVLink | IB | B Error | Final Error | Error Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 197.687 ms | 100.870 ms | 89.1644 ms | 0.674554 ms | 6.62078 ms | skipped | 42.9953% | 1.02922 |
| TQ4 | 116.201 ms | 82.3694 ms | 30.8850 ms | 0.551695 ms | 1.41945 ms | 13.1359% | 44.6350% | 1.06847 |

Compared with the kept fused-FWHT reference:

| Method | Fused FWHT | Fused FWHT + branchless TQ4 bucket | Change |
|---|---:|---:|---:|
| TQ4 | 119.115 ms | 116.201 ms | -2.914 ms |

Compared with the original pre-optimization reference:

| Method | Pre-change | Current kept TQ4 path | Change |
|---|---:|---:|---:|
| TQ4 | 129.813 ms | 116.201 ms | -13.612 ms |

Accuracy stayed unchanged:

```text
Global B Relative Error = 13.1359%
Final Reconstruction Error = 44.6350%
Error Ratio = 1.06847
```

Decision: **keep**. The branchless TQ4 Lloyd-Max lookup improves the full RSVD
TQ4 path and preserves accuracy. Current kept round-1 CUDA optimizations are:

```text
1. shared-memory fused FWHT for rows == 256
2. branchless Lloyd-Max bucket lookup for TQ4
```

## Change: Specialized TQ4 Decode Fast Path

After branchless TQ4 quantization, the microbenchmark showed that the encode
side is no longer dominated by `quantize_bitpack`. The next largest TQ codec
costs are now `decode_event`, `transform`, and `quantize_bitpack`.

This trial targets decode only. The generic fused decode kernel still calls:

```text
bitpack_read_code(codes, idx, bits)
```

even when `bits == 4`. For TQ4, the code is always one 4-bit nibble, so this
trial adds a specialized path:

```text
bitpack_read_code4(...)
column_tq4_lloyd_dequantize_fwht256_apply_add_kernel
column_tq4_lloyd_dequantize_fwht256_apply_store_kernel
```

The specialized kernels are used only when:

```text
rows == 256
mse_bits == 4
```

They keep the same inverse FWHT, Rademacher sign, per-column norm scaling, and
accumulator/store behavior. The only intended change is to remove runtime
bit-width handling and generic centroid boundary checks from the TQ4 decode
path.

Validation plan:

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_tq_kernel_microbench
sbatch run_tq_kernel_microbench.slurm
```

Compare `decode_event` against the current branchless TQ4 reference:

| rows | cols | Current decode_event |
|---:|---:|---:|
| 256 | 16384 | 0.0846 ms |
| 256 | 32768 | 0.1500 ms |

If `decode_event` improves and `relative_error` stays unchanged, run the full
8-GPU validation:

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_cuda_opt_round1
sbatch run_pack4_eval_8gpu.slurm
```

## Result: 2026-06-03

Microbenchmark result:

| rows | cols | decode before | decode with TQ4 fast path | Change |
|---:|---:|---:|---:|---:|
| 256 | 16384 | 0.0846 ms | 0.0970 ms | +14.7% |
| 256 | 32768 | 0.1500 ms | 0.1869 ms | +24.6% |

The TQ4-specific decode path made `decode_event` slower. Accuracy stayed
unchanged, but this does not pass the timing gate.

Likely reason: the original generic fused decode was already cheap enough.
Changing only the 4-bit read from generic word-based bit extraction to a direct
byte/nibble helper did not address the dominant cost, which is the inverse
FWHT, shared-memory synchronization, sign/norm scaling, and accumulator write.
The narrower byte load path may also be less favorable than the original
32-bit word load used by `bitpack_read_code`.

Decision: **fallback**. The TQ4-specialized decode kernels were removed, and
the main path is back to the generic fused decode kernel.

## Change: Branchless Pair-Pack TQ4 Encode

After the branchless bucket lookup, the old microbenchmark-only `pack4_alt`
was revisited. The old pair-pack variant was negative because it still used the
branchy Lloyd-Max bucket lookup. This trial combines:

```text
two TQ4 codes per thread
direct one-byte store
no atomic bitpack write
branchless Lloyd-Max bucket lookup
```

Microbenchmark result:

| rows | cols | Current TQ4 quantize | Branchless pair-pack | Change |
|---:|---:|---:|---:|---:|
| 256 | 16384 | 0.0723 ms | 0.0462 ms | -36.1% |
| 256 | 32768 | 0.1249 ms | 0.0777 ms | -36.2% |

This trial moved the branchless pair-pack kernel into the main TQ4 encode path.
Since the new kernel writes each packed byte directly, the trial also skipped
`clear_codes` for TQ4; other bit widths still used the generic atomic bitpack
path and still cleared their packed output first.

Validation plan:

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_tq_kernel_microbench
sbatch run_tq_kernel_microbench.slurm
```

Then run full RSVD validation:

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_cuda_opt_round1
sbatch run_pack4_eval_8gpu.slurm
```

## Result: 2026-06-03

Microbenchmark after moving branchless pair-pack into the main TQ4 path:

| rows | cols | clear_codes | quantize_bitpack | encode_event_total | relative error |
|---:|---:|---:|---:|---:|---:|
| 256 | 16384 | 0.0027 ms | 0.0470 ms | 0.1813 ms | 0.09717795 |
| 256 | 32768 | 0.0027 ms | 0.0852 ms | 0.3373 ms | 0.09720906 |

This confirmed that the main microbenchmark path did pick up the faster
pair-pack quantization. However, the full RSVD timing got worse:

```text
job_id = 943053
node = gn1217
GPU = 8 x V100
```

| Method | Total Time | GPU Compute | Host/Staging | NVLink | IB | B Error | Final Error | Error Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 194.946 ms | 98.1838 ms | 89.1467 ms | 0.673027 ms | 6.65222 ms | skipped | 42.9953% | 1.02922 |
| TQ4 | 124.048 ms | 76.5978 ms | 44.9617 ms | 0.443911 ms | 1.42149 ms | 13.1359% | 44.6350% | 1.06847 |

Compared with the kept branchless TQ4 reference:

| Method | Kept branchless TQ4 | Branchless pair-pack trial | Change |
|---|---:|---:|---:|
| TQ4 | 116.201 ms | 124.048 ms | +7.847 ms |

Accuracy stayed unchanged, and GPU compute time did decrease. But total runtime
worsened because Host/Staging time increased substantially. This fails the
round-1 keep rule: the optimization must help the full TQ4 pipeline, not only
an isolated kernel measurement.

Decision: **fallback**. The main path was restored to the previous branchless
one-thread-per-coordinate TQ4 quantize kernel with generic atomic bitpack. The
branchless pair-pack kernel remains as a microbenchmark-only negative/diagnostic
result.

## Change: Binary Search Lloyd-Max Bucket Lookup

The original Lloyd-Max TQ encoder finds a bucket by linearly scanning the
codebook boundaries. For TQ4 this means at most 16 comparisons per coordinate.
This trial replaced the linear scan with binary search, so TQ4 should need
around 4 comparisons per coordinate while keeping the same codebook and the same
bit-packed output format.

This is a TQ-only kernel change. It does not touch the shared `none` pipeline.

## Result: 2026-06-02

Run:

```text
job_id = 942640
node = gn1215
GPU = 8 x V100
```

| Method | Total Time | GPU Compute | Host/Staging | NVLink | IB | B Error | Final Error | Error Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 194.683 ms | 100.842 ms | 86.3165 ms | 0.667033 ms | 6.5252 ms | skipped | 42.9953% | 1.02922 |
| TQ4 | 144.438 ms | 96.6036 ms | 45.2358 ms | 0.437803 ms | 1.38937 ms | 13.1359% | 44.6350% | 1.06847 |

Compared with the pre-change reference:

| Method | Pre-change | Binary-search trial | Change |
|---|---:|---:|---:|
| none | 192.389 ms | 194.683 ms | +2.294 ms |
| TQ4 | 129.813 ms | 144.438 ms | +14.625 ms |

Accuracy stayed unchanged, but TQ4 slowed down. For TQ4 the codebook has only
16 levels, so the linear scan is cheap and compiler-friendly. Binary search
adds branchy control flow and likely causes more warp divergence than it saves
in comparisons.

Decision: **fallback**. The code change was reverted; this experiment is kept
only as a negative result.

## Change: Precompute TQ Rademacher Signs

The column-wise TQ path applies a deterministic Rademacher sign before the
Hadamard transform and applies the same sign during decode. The original code
computes the sign inside the CUDA kernel from `(row, col, seed)`.

This trial precomputed those signs into device buffers and changed the TQ
encode/decode kernels to load `+1/-1` from memory instead of recomputing the
hash. The goal was to reduce integer/hash work in the TQ-only part of the
pipeline. It does not change the math, so B error and final error should remain
identical.

## Result: 2026-06-02

Run:

```text
job_id = 942638
node = gn1216
GPU = 8 x V100
```

| Method | Total Time | GPU Compute | Host/Staging | NVLink | IB | B Error | Final Error | Error Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 195.515 ms | 100.331 ms | 87.4884 ms | 0.665378 ms | 6.68389 ms | skipped | 42.9953% | 1.02922 |
| TQ4 | 144.241 ms | 96.792 ms | 44.6541 ms | 0.423961 ms | 1.41695 ms | 13.1359% | 44.6350% | 1.06847 |

Compared with the pre-change reference:

| Method | Pre-change | Precomputed-sign trial | Change |
|---|---:|---:|---:|
| none | 192.389 ms | 195.515 ms | +3.126 ms |
| TQ4 | 129.813 ms | 144.241 ms | +14.428 ms |

Accuracy stayed unchanged, but TQ4 slowed down. The likely reason is that
loading one sign per coordinate adds global-memory traffic in kernels that are
already bandwidth-sensitive. On V100, the original deterministic hash was
cheaper than the extra memory read.

Decision: **fallback**. The code change was reverted; this experiment is kept
only as a negative result.

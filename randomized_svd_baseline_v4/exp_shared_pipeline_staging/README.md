# Shared Pipeline / Staging Breakdown

This experiment is for optimizing the shared data-movement pipeline instead of
only TQ kernels.

The v4 timing summary previously reported a single `Host/Staging Time`. That
was too coarse: it mixed host-device copies, device-host copies, and CPU-side
packing/unpacking/staging work. This experiment adds a breakdown:

```text
Host/Staging Time
  D2H Copy Time
  H2D Copy Time
  Other Host Copy
  CPU Stage Time
```

The payload summary is also split into:

```text
Host-GPU Payload
  D2H Payload
  H2D Payload
```

## Run

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_shared_pipeline_staging
sbatch run_staging_breakdown_8gpu.slurm
```

The script compares:

```text
NONE
TQ4
TQ2
TQ1
```

with the current 8-GPU `64k x 16k` configuration and `repeat=30`.

`TQ2` uses the existing 2-bit Lloyd-Max TQ codebook path. `TQ1` is currently a
sign-only feasibility path: each rotated coordinate stores one sign bit and
decodes to a fixed centroid magnitude. It is useful for first-pass timing and
payload checks; accuracy is not expected to be final yet.

## Result: TQ2/TQ1 Feasibility

Run:

```text
date = 2026-06-03
GPU = 8 x V100
m = 65536
n = 16384
k = 250
oversample = 6
l = 256
repeat = 30
subspace_iter = 1
mpi_size = 1
```

| Method | Total | GPU Compute | NVLink | Subspace Z Reduce | B Reduce/Payload | Compress B | SVD(B) | NVLink Payload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NONE | 175.731 ms | 105.752 ms | 1.76239 ms | 68.7984 ms | 26.9734 ms | 0.000288 ms | 16.6642 ms | 231 MiB |
| TQ4 | 81.44 ms | 80.0832 ms | 1.35636 ms | 5.46171 ms | 4.48481 ms | 4.48226 ms | 16.4078 ms | 147.875 MiB |
| TQ2 | 82.1225 ms | 80.7832 ms | 1.33882 ms | 5.33078 ms | 4.3674 ms | 4.36378 ms | 16.5621 ms | 133.875 MiB |
| TQ1 | 82.6465 ms | 81.3186 ms | 1.32744 ms | 5.10579 ms | 4.13226 ms | 4.1288 ms | 16.5021 ms | 126.875 MiB |

Speedup vs NONE:

| Method | Speedup |
|---|---:|
| TQ4 | 2.16x |
| TQ2 | 2.14x |
| TQ1 | 2.13x |

Payload trend:

```text
TQ4 NVLink payload: 147.875 MiB
TQ2 NVLink payload: 133.875 MiB
TQ1 NVLink payload: 126.875 MiB
```

Interpretation:

- TQ2 and TQ1 both run end-to-end.
- Lower bits reduce the compressed payload, as expected.
- Lower bits do not currently reduce total time, because the run is dominated by
  shared compute phases such as GEMM/QR/SVD plus the TQ encode/decode kernels.
- This result was measured before the TQ1/TQ2 specialized pack kernels were
  added, so it should be treated as the feasibility baseline.
- TQ1/TQ2 now have dedicated encode pack kernels and should be rerun before
  judging them as performance paths.

Decision: **keep as runnable feasibility modes**.

Implemented after this result:

- TQ2 encode now uses `column_tq2_lloyd_quantize_pack4_kernel`, where one
  thread packs four 2-bit codes into one byte.
- TQ1 encode now uses `column_tq1_sign_quantize_pack8_kernel`, where one thread
  packs eight sign bits into one byte.
- TQ2 uses a branchless 4-level bucket lookup.
- TQ1/TQ2 skip `clear_codes` because the pack kernels overwrite the whole
  bit-packed code buffer.

A later pack16/pack32 word-pack attempt was tested but rejected because it made
the TQ2/TQ1 quantize kernels slower by reducing thread-level parallelism.

Implemented after that fallback:

- TQ decode now uses a byte-aligned bit reader for 1-bit, 2-bit, and 4-bit TQ
  payloads.
- This avoids the generic 32-bit cross-word bit reader in the common TQ decode
  path.
- Expected effect: lower `decode_event` in the microbenchmark and slightly
  lower TQ reduce/decompress time in staging. NONE is unchanged.
- TQ4 active encode now uses the pack4 branchless kernel that was previously
  only measured as a microbenchmark alternate path.
- TQ4 also skips `clear_codes`, matching TQ1/TQ2, because its active pack kernel
  overwrites every packed code byte.

Microbenchmark validation:

```text
job_id = 943673
TQ4 quantize_bitpack 16k: 0.0719 ms -> 0.0470 ms
TQ4 quantize_bitpack 32k: 0.1347 ms -> 0.0776 ms
TQ4 roundtrip 16k:        0.2916 ms -> 0.2618 ms
TQ4 roundtrip 32k:        0.5494 ms -> 0.4469 ms
```

Decision at microbenchmark level: **keep**.

End-to-end validation is still needed because the 8-GPU staging run includes
GEMM, QR/TSQR, SVD(B), and NVLink phases that can hide small kernel wins.

## Result: Active TQ4 Pack4 Branchless Encode

Run:

```text
date = 2026-06-03
job_id = 943677
GPU = 8 x V100
m = 65536
n = 16384
k = 250
oversample = 6
l = 256
repeat = 30
subspace_iter = 1
mpi_size = 1
```

| Method | Total | GPU Compute | NVLink | Subspace Z Reduce | B Reduce/Payload | Compress B | SVD(B) | NVLink Payload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NONE | 172.431 ms | 106.457 ms | 1.76376 ms | 65.3037 ms | 26.7060 ms | 0.000266 ms | 16.5296 ms | 231 MiB |
| TQ4 | 83.7724 ms | 82.3943 ms | 1.37769 ms | 5.30512 ms | 4.30628 ms | 4.30374 ms | 16.5257 ms | 147.875 MiB |
| TQ2 | 81.6914 ms | 80.4280 ms | 1.26311 ms | 4.47812 ms | 3.48347 ms | 3.48102 ms | 16.5085 ms | 133.875 MiB |
| TQ1 | 80.2043 ms | 78.9412 ms | 1.26263 ms | 3.81248 ms | 2.81597 ms | 2.81344 ms | 16.4275 ms | 126.875 MiB |

Speedup vs NONE:

| Method | Speedup |
|---|---:|
| TQ4 | 2.06x |
| TQ2 | 2.11x |
| TQ1 | 2.15x |

Comparison with the previous staging run after byte-aligned decode:

| Method | Total before | Total after | B Reduce/Payload before | B Reduce/Payload after | Subspace Z Reduce before | Subspace Z Reduce after |
|---|---:|---:|---:|---:|---:|---:|
| TQ4 | 83.6287 ms | 83.7724 ms | 4.52425 ms | 4.30628 ms | 5.50802 ms | 5.30512 ms |
| TQ2 | 81.8922 ms | 81.6914 ms | 3.51011 ms | 3.48347 ms | 4.51092 ms | 4.47812 ms |
| TQ1 | 81.4480 ms | 80.2043 ms | 2.84430 ms | 2.81597 ms | 3.83556 ms | 3.81248 ms |

Decision: **keep**.

Interpretation:

- The TQ4 compression/reduction phases improve in the expected places:

```text
TQ4 B Reduce/Payload:   4.52425 ms -> 4.30628 ms
TQ4 Compress B:         4.52174 ms -> 4.30374 ms
TQ4 Subspace Z Reduce:  5.50802 ms -> 5.30512 ms
```

- TQ4 total time is almost unchanged because non-compression phases such as
  GEMM, QR/TSQR, and SVD(B) have comparable run-to-run variation.
- TQ2/TQ1 remain stable and slightly improve in total time.
- NONE remains in the same range, so speedup stays around `2x`.

## Change: TQ B Detail Timers

The previous results show that the remaining TQ-specific work is small enough
that total time can be dominated by GEMM/QR/SVD(B) noise. To choose the next CUDA
target more carefully, the program now breaks `Compress B` into:

| Metric | Meaning |
|---|---|
| `TQ B Encode` | local `B_i` column-wise TQ encode on each GPU |
| `TQ B Peer Copy` | compressed code/norm payload movement from worker GPUs to GPU0 |
| `TQ B Decode/Add` | decode compressed `B_i` on GPU0 and accumulate into `B` |

These rows are instrumentation only. They should not change numerical output.
They are expected to be zero for `NONE`.

Next validation:

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_shared_pipeline_staging
sbatch run_staging_breakdown_8gpu.slurm
```

## Result: TQ B Detail Timers

Run:

```text
date = 2026-06-03
GPU = 8 x V100
m = 65536
n = 16384
k = 250
oversample = 6
l = 256
repeat = 30
subspace_iter = 1
mpi_size = 1
```

| Method | Total | B Reduce/Payload | Compress B | TQ B Encode | TQ B Peer Copy | TQ B Decode/Add |
|---|---:|---:|---:|---:|---:|---:|
| NONE | 180.700 ms | 27.2224 ms | 0.000179 ms | 0 ms | 0 ms | 0 ms |
| TQ4 | 87.8514 ms | 4.37280 ms | 4.37037 ms | 1.32401 ms | 0.188404 ms | 2.85481 ms |
| TQ2 | 85.8917 ms | 3.55645 ms | 3.55390 ms | 1.22703 ms | 0.148733 ms | 2.17507 ms |
| TQ1 | 84.0008 ms | 2.87224 ms | 2.86976 ms | 1.23469 ms | 0.147418 ms | 1.48473 ms |

Decision:

- `TQ B Peer Copy` is already small.
- `TQ B Decode/Add` is the largest TQ-specific B component.
- The next CUDA optimization should target GPU0 decode/add, not payload copy.

Candidate next experiment:

- Fuse the eight per-GPU TQ decode/add passes into one GPU0 column kernel.
- The current path decodes each compressed `B_i` sequentially and repeatedly
  updates the global `B` buffer.
- A fused path can read all compressed `B_i` payloads for one column, run the
  inverse transform for each, accumulate in registers, then write `B` once.

## Rejected: Fused TQ B Decode/Add

Implemented and tested a first fused decode/add path for the current staging
condition:

```text
mode=tq
l = 256
mpi_size = 1
check_b_error = off
```

Old B path:

```text
for each GPU g:
  encode B_i on GPU g
  copy compressed B_i payload to GPU0
  decode B_i on GPU0
  add decoded B_i into B
```

New B path:

```text
for each GPU g:
  encode B_i on GPU g
  copy compressed B_i payload to GPU0
  remember GPU0 code/norm pointers

one GPU0 fused kernel:
  for each column:
    for each GPU payload:
      decode one 256-vector
      inverse FWHT
      accumulate into a register
    write B once
```

Expected effect:

- `TQ B Decode/Add` should drop.
- `TQ B Peer Copy` should stay similar.
- `TQ B Encode` should stay similar.
- If the fused kernel increases register/shared-memory pressure too much, it
  may need fallback. Keep only if `B Reduce/Payload` and total time do not
  regress.

Validation:

```text
date = 2026-06-03
GPU = 8 x V100
m = 65536
n = 16384
repeat = 30
subspace_iter = 1
mpi_size = 1
```

| Method | Total | B Reduce/Payload | Compress B | TQ B Encode | TQ B Peer Copy | TQ B Decode/Add |
|---|---:|---:|---:|---:|---:|---:|
| NONE | 186.177 ms | 27.2677 ms | 0.000220 ms | 0 ms | 0 ms | 0 ms |
| TQ4 fused | 87.3523 ms | 3.44889 ms | 2.30217 ms | 1.33590 ms | 0.209980 ms | 0.752217 ms |
| TQ2 fused | 85.7613 ms | 2.70027 ms | 2.13170 ms | 1.23011 ms | 0.168970 ms | 0.728471 ms |
| TQ1 fused | 83.1278 ms | 2.20396 ms | 2.06688 ms | 1.23121 ms | 0.150230 ms | 0.682322 ms |

Compared with the previous non-fused path:

| Method | Total before | Total fused | TQ B Decode/Add before | TQ B Decode/Add fused |
|---|---:|---:|---:|---:|
| TQ4 | 87.8514 ms | 87.3523 ms | 2.85481 ms | 0.752217 ms |
| TQ2 | 85.8917 ms | 85.7613 ms | 2.17507 ms | 0.728471 ms |
| TQ1 | 84.0008 ms | 83.1278 ms | 1.48473 ms | 0.682322 ms |

Decision: **fallback from active code**.

The fused kernel did reduce `TQ B Decode/Add`, but the total-time win was small
and the experiment added a special-case path for `l=256`, one MPI rank, and TQ
without QJL. The active code is back to the simpler per-payload decode/add path
with the detail timers kept. This keeps the benchmark easier to reason about
while we pause CUDA optimization.

## Result: TQ1/TQ2 Specialized Pack Kernels

Run:

```text
date = 2026-06-03
GPU = 8 x V100
m = 65536
n = 16384
k = 250
oversample = 6
l = 256
repeat = 30
subspace_iter = 1
mpi_size = 1
```

| Method | Total | GPU Compute | NVLink | Subspace Z Reduce | B Reduce/Payload | Compress B | SVD(B) | NVLink Payload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NONE | 173.81 ms | 106.66 ms | 1.77663 ms | 66.7166 ms | 26.7684 ms | 0.000284 ms | 16.5446 ms | 231 MiB |
| TQ4 | 83.2369 ms | 81.8666 ms | 1.3695 ms | 5.50714 ms | 4.52768 ms | 4.52481 ms | 16.4496 ms | 147.875 MiB |
| TQ2 | 79.2299 ms | 77.9738 ms | 1.25565 ms | 4.49307 ms | 3.52192 ms | 3.5193 ms | 16.3838 ms | 133.875 MiB |
| TQ1 | 77.2758 ms | 76.0202 ms | 1.25522 ms | 3.76073 ms | 2.80539 ms | 2.80285 ms | 16.3736 ms | 126.875 MiB |

Speedup vs NONE:

| Method | Speedup |
|---|---:|
| TQ4 | 2.09x |
| TQ2 | 2.19x |
| TQ1 | 2.25x |

Change relative to the pre-specialization feasibility run:

| Method | Total before | Total after | Change | B Reduce/Payload before | B Reduce/Payload after | Subspace Z Reduce before | Subspace Z Reduce after |
|---|---:|---:|---:|---:|---:|---:|---:|
| TQ2 | 82.1225 ms | 79.2299 ms | -2.8926 ms | 4.3674 ms | 3.52192 ms | 5.33078 ms | 4.49307 ms |
| TQ1 | 82.6465 ms | 77.2758 ms | -5.3707 ms | 4.13226 ms | 2.80539 ms | 5.10579 ms | 3.76073 ms |

Interpretation:

- The specialized pack kernels turn TQ2/TQ1 from feasibility modes into faster
  performance candidates.
- TQ2 and TQ1 now beat TQ4 in total time for this 8-GPU, single-rank case.
- The gain appears exactly where expected: `Subspace Z Reduce`,
  `B Reduce/Payload`, and `Compress B`.
- Accuracy still needs separate validation. TQ1 in particular is sign-only and
  showed much larger microbenchmark roundtrip error than TQ4/TQ2.

Decision: **keep**.

## How To Read It

Use this run to decide the next shared-pipeline optimization target:

| If this dominates | Likely next target |
|---|---|
| `D2H Copy Time` | reduce GPU-to-host staging, use pinned host buffers, or keep payload on GPU longer |
| `H2D Copy Time` | avoid host-to-GPU restaging after MPI/gather, or batch copies |
| `CPU Stage Time` | remove CPU-side packing/unpacking/accumulation loops |
| `Other/Sync Time` | inspect hidden synchronization and timing attribution |

Shared-pipeline optimizations must still be checked against both `NONE` and
`TQ4`.

Acceptance rule:

- TQ4 absolute time must improve.
- If NONE and TQ4 drop by about the same number of milliseconds, keep it
  because speedup rises.
- If NONE and TQ4 drop by about the same ratio, keep it because speedup stays
  about the same while TQ4 absolute time improves.
- If NONE improves much more than TQ4 and the TQ4-vs-NONE speedup drops
  substantially, park it as a separate shared-baseline optimization instead of
  keeping it in the TQ-focused path.

## Result: Initial Breakdown

Run:

```text
date = 2026-06-03
GPU = 8 x V100
m = 65536
n = 16384
repeat = 30
subspace_iter = 1
```

| Method | Total | Host/Staging | D2H | H2D | CPU Stage | Host-GPU Payload | D2H Payload | H2D Payload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NONE | 197.223 ms | 89.1356 ms | 34.3006 ms | 31.2933 ms | 23.5416 ms | 304 MiB | 152 MiB | 152 MiB |
| TQ4 | 132.263 ms | 46.0908 ms | 10.6423 ms | 31.8756 ms | 3.57285 ms | 200.25 MiB | 44.125 MiB | 156.125 MiB |

Interpretation:

- TQ4 already reduces D2H and CPU staging substantially.
- H2D stays around 31 ms for both NONE and TQ4.
- The dominant H2D source is subspace iteration distributing `Z_global` (`n x l`)
  back to every GPU. For `16k x 256`, this is `16 MiB` per GPU, or `128 MiB`
  across 8 GPUs.

## Change: GPU0-to-GPU Peer Distribution for Subspace Z

Before this change, the pipeline did:

```text
Z_global on GPU0
GPU0 -> host
host -> GPU0
host -> GPU1
...
host -> GPU7
```

The optimized path does:

```text
Z_global on GPU0
GPU0 local device copy
GPU0 -> GPU1 peer copy
...
GPU0 -> GPU7 peer copy
```

For the raw NONE path, `MPI_Allreduce` still produces `Z_global` on host, so the
code performs one host-to-GPU copy to GPU0 first. It no longer performs one H2D
copy per GPU. For compressed TQ4, the reconstructed `Z_global` is already on
GPU0, so it can skip the host round trip entirely.

Expected effect:

| Method | Expected H2D change | Expected NVLink change |
|---|---:|---:|
| NONE | about `-112 MiB` H2D | about `+112 MiB` NVLink |
| TQ4 | about `-128 MiB` H2D plus one removed D2H | about `+112 MiB` NVLink |

Validation:

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_shared_pipeline_staging
sbatch run_staging_breakdown_8gpu.slurm
```

Decision: **pending**. Keep if total time improves for both NONE and TQ4, or if
it improves TQ4 without materially hurting NONE.

## Result: GPU0 Peer Distribution

Run:

```text
date = 2026-06-03
job = 943067 rerun after allocation fix
GPU = 8 x V100
m = 65536
n = 16384
repeat = 30
subspace_iter = 1
```

| Method | Total | GPU Compute | Host/Staging | D2H | H2D | CPU Stage | NVLink | InfiniBand | Host-GPU Payload | D2H Payload | H2D Payload | NVLink Payload | IB Payload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NONE before | 197.223 ms | 100.661 ms | 89.1356 ms | 34.3006 ms | 31.2933 ms | 23.5416 ms | 0.648998 ms | 6.62303 ms | 304 MiB | 152 MiB | 152 MiB | 112 MiB | 36 MiB |
| NONE after | 186.993 ms | 115.163 ms | 63.7492 ms | 32.434 ms | 8.59023 ms | 22.725 ms | 1.38696 ms | 6.43664 ms | 192 MiB | 152 MiB | 40 MiB | 224 MiB | 36 MiB |
| TQ4 before | 132.263 ms | 83.7284 ms | 46.0908 ms | 10.6423 ms | 31.8756 ms | 3.57285 ms | 0.468452 ms | 1.41558 ms | 200.25 MiB | 44.125 MiB | 156.125 MiB | 28.875 MiB | 8.12502 MiB |
| TQ4 after | 117.236 ms | 98.0818 ms | 16.1864 ms | 6.42834 ms | 6.28295 ms | 3.47507 ms | 1.11651 ms | 1.36462 ms | 56.25 MiB | 28.125 MiB | 28.125 MiB | 140.875 MiB | 8.12502 MiB |

Effect:

| Method | Total change | Host/Staging change | H2D change | Host-GPU payload change |
|---|---:|---:|---:|---:|
| NONE | `-10.230 ms` | `-25.386 ms` | `-22.703 ms` | `-112 MiB` |
| TQ4 | `-15.027 ms` | `-29.904 ms` | `-25.593 ms` | `-144 MiB` |

Speedup check:

| Stage | NONE / TQ4 speedup |
|---|---:|
| Initial breakdown | `1.49x` |
| After GPU0 peer distribution | `1.60x` |

Decision: **keep**.

This is a shared-pipeline optimization, but it benefits TQ4 more than NONE in
end-to-end time and increases the TQ4-vs-NONE speedup. It also removes the
original H2D bottleneck that affected both methods.

Interpretation:

- The optimization successfully changed `Z_global` distribution from repeated
  host-to-GPU copies to one GPU0 copy plus peer copies.
- H2D time dropped from about `31 ms` to `6-9 ms`.
- NVLink payload increased because data now moves from GPU0 to the other GPUs.
- TQ4 still keeps far lower host payload than NONE because compressed B and
  compressed subspace payloads are smaller.

Remaining staging cost:

| Method | Main remaining host/staging cost |
|---|---|
| NONE | D2H for raw payloads and CPU staging of raw reductions |
| TQ4 | smaller D2H/H2D compressed payloads plus small CPU pack/unpack staging |

Next candidates:

- Reuse persistent host staging buffers instead of allocating vectors inside
  hot loops.
- Try pinned host buffers for the remaining D2H/H2D payloads.
- Keep peer distribution for `Z_global`; it is clearly beneficial.

## Change: Persistent Pinned Host Staging Buffers

After GPU0 peer distribution, TQ4 still spends about:

```text
D2H Copy Time   6.428 ms
H2D Copy Time   6.283 ms
CPU Stage Time  3.475 ms
```

The remaining hot path still had repeated temporary `std::vector` allocation for
host staging buffers, including:

- raw subspace `Z_i -> host -> Z_local`
- raw `B` reduce buffers
- compressed B code/norm/sign payload buffers
- compressed subspace code/norm/sign payload buffers

This change moves those staging buffers outside the repeat loop and attempts to
allocate them with `cudaMallocHost`, which gives pinned host memory. If pinned
allocation fails, the code falls back to normal pageable `std::vector` storage
instead of failing the run.

Expected effect:

| Method | Expected impact |
|---|---|
| NONE | lower allocation overhead and potentially faster raw D2H/H2D copies |
| TQ4 | lower compressed payload D2H/H2D overhead and less CPU staging noise |

Decision: **validated, but parked**.

Validation:

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_shared_pipeline_staging
sbatch run_staging_breakdown_8gpu.slurm
```

## Result: Persistent Pinned Host Staging Buffers

Run:

```text
date = 2026-06-03
job = 943082 rerun after compile fix
GPU = 8 x V100
m = 65536
n = 16384
repeat = 30
subspace_iter = 1
```

| Method | Total | GPU Compute | Host/Staging | D2H | H2D | CPU Stage | NVLink | InfiniBand | Host-GPU Payload | D2H Payload | H2D Payload | NVLink Payload | IB Payload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NONE before pinned | 186.993 ms | 115.163 ms | 63.7492 ms | 32.434 ms | 8.59023 ms | 22.725 ms | 1.38696 ms | 6.43664 ms | 192 MiB | 152 MiB | 40 MiB | 224 MiB | 36 MiB |
| NONE after pinned | 132.019 ms | 84.1758 ms | 39.512 ms | 14.4199 ms | 4.80961 ms | 20.2825 ms | 1.39259 ms | 6.93849 ms | 192 MiB | 152 MiB | 40 MiB | 224 MiB | 36 MiB |
| TQ4 before pinned | 117.236 ms | 98.0818 ms | 16.1864 ms | 6.42834 ms | 6.28295 ms | 3.47507 ms | 1.11651 ms | 1.36462 ms | 56.25 MiB | 28.125 MiB | 28.125 MiB | 140.875 MiB | 8.12502 MiB |
| TQ4 after pinned | 94.8331 ms | 81.2481 ms | 11.0776 ms | 3.92079 ms | 3.71523 ms | 3.44153 ms | 1.05278 ms | 1.4542 ms | 56.25 MiB | 28.125 MiB | 28.125 MiB | 140.875 MiB | 8.12502 MiB |

Effect:

| Method | Total change | Host/Staging change | D2H change | H2D change | CPU Stage change |
|---|---:|---:|---:|---:|---:|
| NONE | `-54.974 ms` | `-24.237 ms` | `-18.014 ms` | `-3.781 ms` | `-2.443 ms` |
| TQ4 | `-22.403 ms` | `-5.108 ms` | `-2.508 ms` | `-2.568 ms` | `-0.034 ms` |

Decision: **fallback from the TQ-focused path; keep documented as a separate
shared-baseline optimization candidate**.

This is a valid shared-pipeline optimization, but it benefits NONE much more in
absolute time. The TQ4 total time improves, but the headline speedup drops:

| Stage | NONE / TQ4 speedup |
|---|---:|
| Initial breakdown | `1.49x` |
| After GPU0 peer distribution | `1.60x` |
| After pinned staging | `1.39x` |

The speedup ratio falls after pinned staging because NONE had much more raw
pageable D2H traffic to improve. Since the current goal is to preserve and
explain the TQ/TQ4 advantage over NONE, this change was reverted from the main
TQ-focused code path after validation.

Current kept shared-pipeline optimizations:

1. GPU0-to-GPU peer distribution for subspace `Z_global`.

Parked shared-baseline candidate:

1. Persistent host staging buffers with pinned-memory fallback.

TQ-focused baseline after pinned-staging fallback was the GPU0 peer-distribution
result:

```text
NONE Total   186.993 ms
TQ4 Total    117.236 ms
Speedup      1.60x
```

Next candidates should include both TQ-specific kernels and shared-pipeline
changes, as long as they satisfy the acceptance rule above. Avoid touching QJL
in this round.

## Change: Single-Rank Compressed Collective Bypass

The 8-GPU staging experiment uses one MPI rank:

```text
mpi_size = 1
local_gpus_per_rank = 8
```

Before this change, the TQ path still performed the cross-rank compressed
collective sequence even though there was only one rank:

```text
compressed B or Z on GPU0
GPU0 -> host compressed payload
MPI Gather/Allgather with one rank
host -> GPU0 compressed payload
decompress back into GPU0 buffer
```

With one MPI rank this is mathematically a no-op. The local GPU reduction has
already produced the correct GPU0 result:

- `d_mpi_Z_local` already contains the local reduced subspace `Z`.
- `d_B` already contains the local reduced `B`.

The optimized TQ path now bypasses the compressed cross-rank collective when
`mpi.size == 1`:

```text
subspace Z: d_mpi_Z_local -> d_mpi_Z_global by device-to-device copy
B: keep d_B as-is
```

This is intentionally TQ-focused:

| Method | Change |
|---|---|
| NONE | unchanged |
| TQ4 | skip redundant single-rank compressed B/Z MPI payload staging |

Expected effect:

- TQ4 total time should drop.
- TQ4 Host/Staging should drop.
- TQ4 InfiniBand payload should drop because there is no real inter-rank
  payload in the one-rank run.
- NONE should stay about the same.
- TQ4-vs-NONE speedup should increase.

## Result: Parallel TQ Encode Launch for Z and B

Run:

```text
date = 2026-06-03
GPU = 8 x V100
m = 65536
n = 16384
repeat = 30
subspace_iter = 1
mpi_size = 1
```

| Method | Total | GPU Compute | NVLink | Subspace Z Reduce | B Reduce/Payload | Compress B |
|---|---:|---:|---:|---:|---:|---:|
| TQ4 before parallel encode | 88.2217 ms | 86.8796 ms | 1.34125 ms | 5.57692 ms | 4.59961 ms | 4.59707 ms |
| TQ4 after parallel encode | 88.6373 ms | 87.2605 ms | 1.37628 ms | 5.65834 ms | 4.65707 ms | 1.58244 ms |

Decision: **fallback**.

Although the `Compress B` sub-timer dropped, the total TQ4 time got worse and
both TQ-specific aggregate phases increased:

```text
TQ4 Total:             +0.4156 ms
Subspace Z Reduce:     +0.08142 ms
B Reduce/Payload:      +0.05746 ms
NVLink:                +0.03503 ms
```

The likely reason is that the encode kernels are already short enough that
splitting the pipeline into a separate "launch all encode, sync all, then
gather/decode" phase adds synchronization and scheduling overhead without
reducing the critical path. The B-side timer improvement only changes where the
time is attributed; it does not improve the whole B reduce/payload phase.

The main code was restored to the original sequential encode/copy/decode order
for subspace TQ. The B parallel fast path is parked/disabled.

Decision: **fallback from the active path**.

Validation:

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_shared_pipeline_staging
sbatch run_staging_breakdown_8gpu.slurm
```

## Result: Skip Host Subspace Z Buffer

Run:

```text
date = 2026-06-03
GPU = 8 x V100
m = 65536
n = 16384
repeat = 30
subspace_iter = 1
mpi_size = 1
```

| Method | Total | GPU Compute | Host/Staging | D2H | H2D | CPU Stage | NVLink | InfiniBand | Host-GPU Payload | D2H Payload | H2D Payload | NVLink Payload | IB Payload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NONE before | 187.106 ms | 115.657 ms | 63.0258 ms | 31.6874 ms | 8.57589 ms | 22.7625 ms | 1.38755 ms | 6.69616 ms | 192 MiB | 152 MiB | 40 MiB | 224 MiB | 36 MiB |
| NONE after | 187.391 ms | 116.685 ms | 62.5327 ms | 31.854 ms | 8.61626 ms | 22.0624 ms | 1.38337 ms | 6.48687 ms | 192 MiB | 152 MiB | 40 MiB | 224 MiB | 36 MiB |
| TQ4 before | 104.989 ms | 95.217 ms | 7.65015 ms | 2.38039 ms | 1.77019 ms | 3.49957 ms | 1.09539 ms | 0.600872 ms | 16 MiB | 8 MiB | 8 MiB | 140.875 MiB | 4 MiB |
| TQ4 after | 92.4588 ms | 85.8161 ms | 5.05196 ms | 2.30245 ms | 1.72657 ms | 1.02294 ms | 1.01475 ms | 0.575539 ms | 16 MiB | 8 MiB | 8 MiB | 140.875 MiB | 4 MiB |

Effect:

| Method | Total change | Host/Staging change | CPU Stage change | Payload change |
|---|---:|---:|---:|---:|
| NONE | `+0.285 ms` | `-0.493 ms` | `-0.700 ms` | `0 MiB` |
| TQ4 | `-12.530 ms` | `-2.598 ms` | `-2.477 ms` | `0 MiB` |

Speedup check:

| Stage | NONE / TQ4 speedup |
|---|---:|
| Initial breakdown | `1.49x` |
| After GPU0 peer distribution | `1.60x` |
| After single-rank compressed collective bypass | `1.78x` |
| After skipping host subspace Z buffer | `2.03x` |

Decision: **keep**.

This is TQ-path-only in the single-rank compressed bypass case. It removes
unneeded host allocation/clear work for `Z`, so NONE remains effectively
unchanged while TQ4 improves by about `12.5 ms`.

Current TQ-focused baseline:

```text
NONE Total   187.391 ms
TQ4 Total     92.4588 ms
Speedup       2.03x
```

## Change: Single-Rank Device TSQR Metadata Path

The remaining TQ4 host payload after the previous optimizations is:

```text
D2H Payload  8 MiB
H2D Payload  8 MiB
```

This matches the two TSQR metadata exchanges in the current configuration:

1. initial TSQR for `Y = A Omega`
2. subspace-iteration TSQR for `Y = A Z`

Before this change, each TSQR stage did:

```text
R_i on each GPU
GPU -> host
host packs Rstack
host -> GPU0
GPU0 QR(Rstack)
GPU0 -> host
host slices T_i
host -> each GPU
```

When `mpi_size=1`, there is no cross-rank TSQR metadata exchange. The optimized
path keeps the metadata on GPUs:

```text
R_i on each GPU
GPU_i -> GPU0 Rstack by GPU peer copies
GPU0 QR(Rstack)
GPU0 Rstack row-blocks -> GPU_i T_i by GPU peer copies
```

This is a shared-pipeline optimization, but it targets the remaining TSQR
metadata path rather than broad pageable host copies.

Expected effect:

- TQ4 Host-GPU payload should drop from `16 MiB` toward `0 MiB`.
- NONE should also lose the TSQR host payload, but its raw subspace/B payloads
  are much larger, so the relative speedup should not collapse.
- NVLink payload should increase by about the moved TSQR metadata size.
- Keep if TQ4 absolute time improves and speedup stays about the same or
  improves.

Decision: **fallback after validation**.

Validation:

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_shared_pipeline_staging
sbatch run_staging_breakdown_8gpu.slurm
```

## Result: Single-Rank Device TSQR Metadata Path

Run:

```text
date = 2026-06-03
GPU = 8 x V100
m = 65536
n = 16384
repeat = 30
subspace_iter = 1
mpi_size = 1
```

| Method | Total | GPU Compute | Host/Staging | D2H | H2D | CPU Stage | NVLink | InfiniBand | Host-GPU Payload | D2H Payload | H2D Payload | NVLink Payload | IB Payload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NONE before TSQR peer path | 187.391 ms | 116.685 ms | 62.5327 ms | 31.854 ms | 8.61626 ms | 22.0624 ms | 1.38337 ms | 6.48687 ms | 192 MiB | 152 MiB | 40 MiB | 224 MiB | 36 MiB |
| NONE after TSQR peer path | 335.611 ms | 143.296 ms | 59.1974 ms | 30.8831 ms | 6.91654 ms | 21.3978 ms | 127.006 ms | 5.86417 ms | 176 MiB | 144 MiB | 32 MiB | 232 MiB | 32 MiB |
| TQ4 before TSQR peer path | 92.4588 ms | 85.8161 ms | 5.05196 ms | 2.30245 ms | 1.72657 ms | 1.02294 ms | 1.01475 ms | 0.575539 ms | 16 MiB | 8 MiB | 8 MiB | 140.875 MiB | 4 MiB |
| TQ4 after TSQR peer path | 238.025 ms | 114.124 ms | 0 ms | 0 ms | 0 ms | 0 ms | 123.901 ms | 0 ms | 0 MiB | 0 MiB | 0 MiB | 148.875 MiB | 0 MiB |

Effect:

| Method | Total change | Host/Staging change | NVLink change | Host-GPU payload change |
|---|---:|---:|---:|---:|
| NONE | `+148.220 ms` | `-3.335 ms` | `+125.623 ms` | `-16 MiB` |
| TQ4 | `+145.566 ms` | `-5.052 ms` | `+122.886 ms` | `-16 MiB` |

Decision: **fallback**.

The idea removed the remaining TSQR host payload, but it made the run much
slower. The problem is that `Rstack` and `T_i` are column-major row slices:
copying each small row segment as an individual peer copy creates many tiny
GPU peer-copy calls. The payload is small, but the API/synchronization overhead
is large, so NVLink time jumps from about `1 ms` to more than `120 ms`.

This is not a good TQ-focused optimization in the current implementation. The
main code was restored to the previous host-staged TSQR metadata path.

Current kept TQ-focused baseline remains:

```text
NONE Total   187.391 ms
TQ4 Total     92.4588 ms
Speedup       2.03x
```

## Change: Packed Single-Rank Device TSQR Metadata Path

The failed TSQR peer-copy attempt showed the right payload target but the wrong
copy shape. It removed the remaining TQ4 host payload, but it copied many tiny
row slices through peer-copy calls, which made NVLink/API overhead dominate.

The revised path keeps the same goal but changes the data movement:

```text
R_i on GPU_i
GPU_i -> GPU0 contiguous l*l R_i copy
GPU0 kernel packs R_i into strided Rstack
GPU0 QR(Rstack)
GPU0 kernel slices each T_i into a contiguous l*l buffer
GPU0 -> GPU_i contiguous l*l T_i copy
```

Compared with the failed path:

| Path | Copy shape | Expected behavior |
|---|---|---|
| Failed TSQR peer path | many tiny row-slice peer copies | high API/sync overhead |
| Packed TSQR peer path | one contiguous `l*l` copy per GPU per direction, plus pack/slice kernels | much lower peer-copy overhead |

Expected effect:

- TQ4 `Host-GPU Payload` should again move from `16 MiB` toward `0 MiB`.
- TQ4 `Host/Staging Time` should drop from about `5 ms`.
- NVLink time should increase, but it must stay near a few ms, not `120+ ms`.
- NONE may also improve because TSQR metadata is shared pipeline, but keep only
  if TQ4 absolute time improves and speedup does not collapse.

Validation:

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_shared_pipeline_staging
sbatch run_staging_breakdown_8gpu.slurm
```

Decision: **keep after validation**.

## Result: Packed Single-Rank Device TSQR Metadata Path

Run:

```text
date = 2026-06-03
GPU = 8 x V100
m = 65536
n = 16384
repeat = 30
subspace_iter = 1
mpi_size = 1
```

| Method | Total | GPU Compute | Host/Staging | D2H | H2D | CPU Stage | NVLink | InfiniBand | Host-GPU Payload | D2H Payload | H2D Payload | NVLink Payload | IB Payload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NONE before packed TSQR | 191 ms | 116.574 ms | 66.2511 ms | 34.5834 ms | 8.75642 ms | 22.9112 ms | 1.42443 ms | 6.60086 ms | 192 MiB | 152 MiB | 40 MiB | 224 MiB | 36 MiB |
| NONE after packed TSQR | 173.969 ms | 107.786 ms | 58.0655 ms | 30.2066 ms | 6.83072 ms | 21.0282 ms | 1.76205 ms | 6.00606 ms | 176 MiB | 144 MiB | 32 MiB | 231 MiB | 32 MiB |
| TQ4 before packed TSQR | 92.3822 ms | 85.803 ms | 5.0207 ms | 2.27452 ms | 1.71587 ms | 1.03031 ms | 1.00241 ms | 0.555774 ms | 16 MiB | 8 MiB | 8 MiB | 140.875 MiB | 4 MiB |
| TQ4 after packed TSQR | 84.2393 ms | 82.868 ms | 0 ms | 0 ms | 0 ms | 0 ms | 1.37079 ms | 0 ms | 0 MiB | 0 MiB | 0 MiB | 147.875 MiB | 0 MiB |

Effect:

| Method | Total change | Host/Staging change | GPU Compute change | NVLink change | Host-GPU payload change |
|---|---:|---:|---:|---:|---:|
| NONE | `-17.031 ms` | `-8.186 ms` | `-8.788 ms` | `+0.338 ms` | `-16 MiB` |
| TQ4 | `-8.143 ms` | `-5.021 ms` | `-2.935 ms` | `+0.368 ms` | `-16 MiB` |

Speedup check:

| Stage | NONE / TQ4 speedup |
|---|---:|
| Before packed TSQR | `2.07x` |
| After packed TSQR | `2.06x` |

Decision: **keep**.

This matches the goal for shared-pipeline optimization: TQ4 absolute time
improves, and speedup stays essentially unchanged. The packed device path also
removes the last TQ4 host-staging payload:

```text
TQ4 Host-GPU Payload: 16 MiB -> 0 MiB
TQ4 Host/Staging:      5.0207 ms -> 0 ms
TQ4 Total:            92.3822 ms -> 84.2393 ms
```

The earlier failed TSQR peer path was slow because it used many tiny peer
copies. This packed path is fast because it moves contiguous `l*l` blocks and
uses small GPU kernels only for the strided pack/slice work.

Current kept TQ-focused baseline:

```text
NONE Total   173.969 ms
TQ4 Total     84.2393 ms
Speedup       2.06x
```

Next target:

- TQ4 is now dominated by GPU-side compute (`82.868 ms` out of `84.2393 ms`).
- Host/staging and inter-rank staging are no longer useful targets for the
  single-rank TQ4 path.
- The next experiment should add a GPU phase breakdown for TQ4 to separate
  GEMM/QR/TSQR/SVD from TQ encode/decode/reduce kernels.

## Change: Algorithm Phase Summary

After packed TSQR, TQ4 no longer has measurable host/staging time in the
single-rank 8-GPU experiment. The next bottleneck is inside GPU-side algorithm
time, so the program now prints an additional summary:

```text
Algorithm Phase Summary
  Local Projection Y
  Local QR
  TSQR Reduce
  Form Distributed Q
  Subspace Iteration
    Subspace Z GEMM
    Subspace Z Reduce
    Subspace Qbar GEMM
    Subspace QR/TSQR
    Compress Subspace
  Build/Reduce B
    B GEMM
    B Reduce/Payload
    Compress B
  SVD(B)
  Form Distributed U
```

How to use it:

| If this phase dominates TQ4 | Next optimization direction |
|---|---|
| `Subspace Z GEMM` or `Subspace Qbar GEMM` | shared GEMM path; compare carefully against NONE |
| `Subspace Z Reduce` | optimize TQ compression/reconstruction of `Z = A^T Q` |
| `Subspace QR/TSQR` | revisit local QR, packed TSQR, or distributed Q formation |
| `B GEMM` | shared GEMM path for `B_i = Q_i^T A_i`; compare carefully against NONE |
| `B Reduce/Payload` and `Compress B` | optimize TQ encode/decode/reduce kernels or compressed payload movement |
| `TSQR Reduce` | revisit GPU TSQR packing/QR/slicing |
| `Local Projection Y` or `Form Distributed Q` | this is shared GEMM/QR pipeline, so compare carefully against NONE |
| `SVD(B)` | cuSOLVER small dense SVD bottleneck; likely harder to improve without changing algorithm |

Note: `Compress Subspace` only reports the second-stage MPI/rank-level
subspace compression path. In the single-rank TQ bypass, the important local
TQ work is included in `Subspace Z Reduce`, so use that row for this 8-GPU
single-node experiment.

## Parked Attempt: Parallel TQ Encode Launch for Z and B

The detailed phase breakdown showed the main TQ-specific costs:

```text
TQ4 Subspace Z Reduce    5.57692 ms
TQ4 B Reduce/Payload     4.59961 ms
TQ4 Compress B           4.59707 ms
```

Before this change, the code processed TQ payloads one GPU at a time:

```text
GPU0 encode -> copy/decode
GPU1 encode -> copy/decode
...
GPU7 encode -> copy/decode
```

That serializes encode work that could run independently on all GPUs. The
attempted path split it into two phases:

```text
all GPUs launch transpose/encode
sync all GPUs
GPU0 gathers compressed payloads and decodes/adds
```

Applied to:

- subspace TQ reduce for `Z = A^T Q`
- B TQ reduce for `B_i = Q_i^T A_i`

The B-side fast path was only used for the normal benchmark case:

```text
compress_b_mode = tq
check_b_error = false
global diagnostic B metric = false
```

It does not change the QJL path.

Expected effect was:

- `Subspace Z Reduce` should drop.
- `B Reduce/Payload` and `Compress B` should drop.
- NONE should be unchanged.
- TQ4 total time should improve with no speedup penalty.

Validation result: **fallback**. See "Result: Parallel TQ Encode Launch for Z
and B" above for the measured slowdown. The subspace path was restored to
sequential encode/copy/decode, and the B fast path is disabled in code.

Decision: **pending validation**.

Validation:

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_shared_pipeline_staging
sbatch run_staging_breakdown_8gpu.slurm
```

## Current Kept Baseline

Latest kept run after the useful staging and TQ-path changes:

```text
date = 2026-06-03
GPU = 8 x V100
m = 65536
n = 16384
repeat = 30
subspace_iter = 1
mpi_size = 1
```

| Method | Total | GPU Compute | Host/Staging | D2H | H2D | CPU Stage | NVLink | InfiniBand | Host-GPU Payload | D2H Payload | H2D Payload | NVLink Payload | IB Payload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NONE | 181.47 ms | 111.428 ms | 61.6596 ms | 31.3881 ms | 8.56016 ms | 21.7114 ms | 1.34032 ms | 6.66916 ms | 192 MiB | 152 MiB | 40 MiB | 224 MiB | 36 MiB |
| TQ4 | 87.4631 ms | 86.1011 ms | 0 ms | 0 ms | 0 ms | 0 ms | 1.36144 ms | 0 ms | 0 MiB | 0 MiB | 0 MiB | 147.875 MiB | 0 MiB |

Speedup:

```text
NONE / TQ4 = 2.07x
```

TQ4 detailed phase summary:

| Phase | Time |
|---|---:|
| Local Projection Y | 5.35435 ms |
| Local QR | 6.91725 ms |
| TSQR Reduce | 2.50664 ms |
| Form Distributed Q | 0.658354 ms |
| Subspace Iteration | 42.5696 ms |
| Subspace Z GEMM | 5.4404 ms |
| Subspace Z Reduce | 5.59039 ms |
| Subspace Qbar GEMM | 21.7385 ms |
| Subspace QR/TSQR | 9.80005 ms |
| Build/Reduce B | 10.1943 ms |
| B GEMM | 5.59732 ms |
| B Reduce/Payload | 4.59252 ms |
| Compress B | 4.58979 ms |
| SVD(B) | 19.262 ms |

Current interpretation:

- The useful TQ-specific staging work already removed host staging for TQ4 in
  this single-node case.
- The main remaining TQ-specific costs are `Subspace Z Reduce` and
  `B Reduce/Payload`.
- The largest remaining total phases are shared pipeline phases:
  `Subspace Qbar GEMM` and `SVD(B)`.

## Kept: Packed Single-Rank Device TSQR Metadata Path

Earlier TSQR optimization attempted many small peer copies and was rejected:

```text
NVLink time jumped above 120 ms
TQ4 total time became much worse
```

The kept version uses contiguous GPU0 staging buffers instead:

```text
R_i on each GPU
  -> one contiguous l x l peer copy per GPU into GPU0 stage buffer
  -> GPU0 pack kernel writes into R stack
  -> TSQR QR on GPU0
  -> GPU0 slice kernel extracts T_i blocks
  -> one contiguous l x l peer copy per nonzero GPU
```

Why this works:

- It keeps TSQR metadata on device for `mpi_size=1`.
- It avoids the CPU pack/unpack loop.
- It avoids thousands of tiny row peer copies.

Decision: **keep**.

## Rejected: Tiny Peer-Copy TSQR Metadata Path

The rejected version tried to move TSQR metadata directly row-by-row between
GPUs. It looked attractive because it avoided host staging, but it created many
small peer copies.

Observed effect:

```text
NVLink time: about 1 ms -> more than 120 ms
TQ4 total time: about 92 ms -> about 238 ms
```

Decision: **fallback**.

## Rejected: Parallel TQ Encode Launch for Z and B

The rejected version launched all TQ encode kernels first and gathered payloads
afterward. It reduced the local `Compress B` sub-timer, but total time was
worse:

```text
TQ4 total: 88.2217 ms -> 88.6373 ms
Subspace Z Reduce: 5.57692 ms -> 5.65834 ms
B Reduce/Payload: 4.59961 ms -> 4.65707 ms
```

Decision: **fallback**.

The B-side parallel path is parked in code behind:

```cpp
const bool use_parallel_tq_b_encode = false;
```

## Rejected: Direct Subspace Z Source for Single-Rank TQ Bypass

In the single-rank TQ subspace path, decompression already produces `Z` on GPU0:

```text
d_mpi_ZT_local -> transpose -> d_mpi_Z_local
```

Before this change, the code copied it again:

```text
d_mpi_Z_local -> d_mpi_Z_global
```

Then the next GEMM distributed `d_mpi_Z_global` to each GPU.

The attempted path skipped that extra device-to-device copy:

```text
d_mpi_Z_local is used directly as the source for the next GEMM
```

Expected effect:

- TQ4 only.
- NONE unchanged.
- Small improvement in `Subspace Z Reduce` or total GPU compute.
- Payload table unchanged, because this is an internal GPU0 device copy.

Validation result:

| Method | Total | GPU Compute | NVLink | Subspace Z Reduce | B Reduce/Payload |
|---|---:|---:|---:|---:|---:|
| TQ4 before | 87.4631 ms | 86.1011 ms | 1.36144 ms | 5.59039 ms | 4.59252 ms |
| TQ4 after | 87.7619 ms | 86.1959 ms | 1.5655 ms | 5.62167 ms | 4.6341 ms |

Decision: **fallback**.

The direct source path did not improve the TQ4 critical path. It removed one
internal GPU0 device copy in theory, but the measured run was slightly slower:

```text
TQ4 Total:             +0.2988 ms
Subspace Z Reduce:     +0.03128 ms
B Reduce/Payload:      +0.04158 ms
NVLink:                +0.20406 ms
```

The likely explanation is that the skipped copy was not on the critical path,
or that using the alternate source pointer changed copy/scheduling order enough
to lose more than the saved D2D copy. The main code was restored to the previous
`d_mpi_Z_local -> d_mpi_Z_global` device-copy path.

## Result: Single-Rank Compressed Collective Bypass

Run:

```text
date = 2026-06-03
GPU = 8 x V100
m = 65536
n = 16384
repeat = 30
subspace_iter = 1
mpi_size = 1
```

| Method | Total | GPU Compute | Host/Staging | D2H | H2D | CPU Stage | NVLink | InfiniBand | Host-GPU Payload | D2H Payload | H2D Payload | NVLink Payload | IB Payload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NONE before bypass | 186.993 ms | 115.163 ms | 63.7492 ms | 32.434 ms | 8.59023 ms | 22.725 ms | 1.38696 ms | 6.43664 ms | 192 MiB | 152 MiB | 40 MiB | 224 MiB | 36 MiB |
| NONE after bypass | 187.106 ms | 115.657 ms | 63.0258 ms | 31.6874 ms | 8.57589 ms | 22.7625 ms | 1.38755 ms | 6.69616 ms | 192 MiB | 152 MiB | 40 MiB | 224 MiB | 36 MiB |
| TQ4 before bypass | 117.236 ms | 98.0818 ms | 16.1864 ms | 6.42834 ms | 6.28295 ms | 3.47507 ms | 1.11651 ms | 1.36462 ms | 56.25 MiB | 28.125 MiB | 28.125 MiB | 140.875 MiB | 8.12502 MiB |
| TQ4 after bypass | 104.989 ms | 95.217 ms | 7.65015 ms | 2.38039 ms | 1.77019 ms | 3.49957 ms | 1.09539 ms | 0.600872 ms | 16 MiB | 8 MiB | 8 MiB | 140.875 MiB | 4 MiB |

Effect:

| Method | Total change | Host/Staging change | D2H change | H2D change | IB payload change |
|---|---:|---:|---:|---:|---:|
| NONE | `+0.113 ms` | `-0.723 ms` | `-0.747 ms` | `-0.014 ms` | `0 MiB` |
| TQ4 | `-12.247 ms` | `-8.536 ms` | `-4.048 ms` | `-4.513 ms` | `-4.125 MiB` |

Speedup check:

| Stage | NONE / TQ4 speedup |
|---|---:|
| Initial breakdown | `1.49x` |
| After GPU0 peer distribution | `1.60x` |
| After single-rank compressed collective bypass | `1.78x` |

Decision: **keep**.

This is a TQ-focused pipeline optimization. It removes compressed collective
staging that is redundant when `mpi_size=1`, so NONE is effectively unchanged
while TQ4 absolute time and speedup both improve.

## Change: Skip Host Subspace Z Buffer in Single-Rank TQ Bypass

After the single-rank compressed collective bypass, the TQ subspace path no
longer needs host-side `Z` buffers:

```text
d_mpi_Z_local on GPU0 -> d_mpi_Z_global on GPU0
```

However, the code still allocated host `h_Z_local/h_Z_global` and cleared
`h_Z_local` every repeat. For `n=16k, l=256`, each host Z buffer is `16 MiB`.

The new path only allocates and clears host `Z` when it is actually needed:

| Method/path | Host Z needed? |
|---|---|
| NONE/raw subspace collective | yes |
| multi-rank compressed subspace collective | yes |
| single-rank TQ compressed bypass | no |

Expected effect:

- TQ4 CPU Stage should drop slightly.
- NONE should be unchanged.
- Payload numbers should be unchanged.
- TQ4-vs-NONE speedup should increase slightly or stay the same.

Decision: **pending validation**.

Validation:

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_shared_pipeline_staging
sbatch run_staging_breakdown_8gpu.slurm
```

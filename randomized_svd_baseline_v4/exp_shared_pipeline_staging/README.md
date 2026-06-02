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
```

with the current 8-GPU `64k x 16k` configuration and `repeat=30`.

## How To Read It

Use this run to decide the next shared-pipeline optimization target:

| If this dominates | Likely next target |
|---|---|
| `D2H Copy Time` | reduce GPU-to-host staging, use pinned host buffers, or keep payload on GPU longer |
| `H2D Copy Time` | avoid host-to-GPU restaging after MPI/gather, or batch copies |
| `CPU Stage Time` | remove CPU-side packing/unpacking/accumulation loops |
| `Other/Sync Time` | inspect hidden synchronization and timing attribution |

Shared-pipeline optimizations must still be checked against both `NONE` and
`TQ4`. If an optimization helps `NONE` much more than `TQ4`, keep it out of the
TQ-focused round and document it as a separate shared-baseline improvement.
Fallback rule: if a shared-pipeline optimization lowers the TQ4-vs-NONE speedup
substantially, do not keep it in the TQ-focused path even if TQ4 absolute time
also improves.

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

Current TQ-focused baseline after fallback is the GPU0 peer-distribution result:

```text
NONE Total   186.993 ms
TQ4 Total    117.236 ms
Speedup      1.60x
```

Next candidates:

- Prioritize TQ-specific compute/kernel work before broad pageable-copy
  optimizations.
- If revisiting shared pipeline later, evaluate pinned staging in a separate
  shared-baseline round.
- Avoid touching QJL in this round.

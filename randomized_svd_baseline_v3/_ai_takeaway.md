# AI Takeaway — randomized_svd_baseline_v3/

## Purpose

**Final multi-node benchmark**. This is the MPI multi-node extension of v2.
It is the primary location for claiming cross-node speedup from compressed
MPI collectives.

Key addition over v2: the cross-node `B` reduction uses **compressed gather/decode**
(not FP32 MPI_Reduce) when `--compress-b-mode tq/lowbit/tq-qjl` is set.
This is where the ~19–22% speedup claim comes from.

Current limit: **16 GPUs** (2 nodes × 8 GPUs) due to Taiwania 2 QoS policy.

---

## Files

| File | Description |
|------|-------------|
| `randomized_svd_multigpu_v3.cu` | Main source. MPI + CUDA. One rank per node, all local GPUs per rank. |
| `Makefile` | Requires MPI modules. `module load ucx/1.14.1 openmpi/5.0.2_ucx1.14.1_cuda12.3`. |
| `README.md` | Current results, communication model, build/run instructions. |
| `run_randomized_svd_multigpu_v3_tq_bit_curve_16gpu.slurm` | Main 16-GPU sweep (m=32768). |
| `run_randomized_svd_multigpu_v3_tq_bit_curve_16gpu_large.slurm` | Large 16-GPU sweep (m=65536). |
| `run_randomized_svd_multigpu_v3_smoke_16gpu.slurm` | Quick smoke test: one mode, few repeats. |
| `run_multinode_gpu_visibility_probe_32gpu.slurm` | 32-GPU probe (pending QoS approval). |

---

## Key Concepts

### MPI Layout

```
MPI rank 0  →  node 0, GPUs 0..7
MPI rank 1  →  node 1, GPUs 0..7
global GPUs = ranks × GPUs_per_rank = 16
```

Global matrix is row-split across all 16 GPUs:
```
A = [A_0; A_1; ...; A_15]
```

### Pipeline Flow (v3 additions highlighted)

```
[same per-GPU TSQR pipeline as v2]
↓
Each rank: gather local R_i to host
↓
MPI rank 0: TSQR on stacked R_i → broadcast T_i   [MPI collective]
↓
Each GPU: B_i = Q_i^T A_i
↓
Each rank: node-local B_i compression/reconstruction
↓
Each rank: compress rank-local B                   [TurboQuant kernels]
↓
MPI gather compressed B payloads + metadata        [MPI_Gather compressed]  ← KEY
↓
Rank 0: decode/add each payload on GPU 0           [fused decode-add kernel]
↓
Rank 0: SVD(B)
```

### Two Compression Levels

The v3 output reports two compression stages:

| Metric prefix | What it measures |
|---------------|-----------------|
| `reduce_B_*` | Node-local per-GPU B_i compression (same as v2) |
| `mpi_B_*` | Cross-node rank-level B compression sent through MPI |

For the headline result (none vs TQ 4-bit), `mpi_B_payload_MiB` is the key
number: the compressed payload actually crossing the network.

### MPI Communication Modes

| `--compress-b-mode` | MPI B collective | Comment |
|---------------------|-----------------|---------|
| `none` | `fp32_reduce` (MPI_Reduce FP32) | Full FP32 B crosses network |
| `tq` / `lowbit` / `tq-qjl` | `compressed_gather_decode` | Only compressed payload crosses |

---

## How to Build & Run

```bash
module load cuda/12.8
module load ucx/1.14.1
module load openmpi/5.0.2_ucx1.14.1_cuda12.3
cd randomized_svd_baseline_v3
make

# Main 16-GPU large run (headline result)
sbatch run_randomized_svd_multigpu_v3_tq_bit_curve_16gpu_large.slurm

# Smoke test (quick sanity check)
sbatch run_randomized_svd_multigpu_v3_smoke_16gpu.slurm
```

---

## Current Headline Results

### Base run (m=32768, n=8192, k=256, 16 GPUs, repeat=5)

| Mode | Warm Pipeline | Node-local B | MPI B | B Error |
|------|--------------|-------------|-------|---------|
| none | 64.56 ms | 160 MiB | 20 MiB | — |
| TQ 4-bit | 50.09 ms (**−22.4%**) | 32 MiB | 4 MiB | 0.175 |
| TQ 2-bit | 47.17 ms (**−26.9%**) | 16 MiB | 2 MiB | 0.953 |

### Large run (m=65536, n=16384, k=256, 16 GPUs, repeat=20) — after all optimizations

| Mode | Warm Pipeline | Node-local B | MPI B | B Error |
|------|--------------|-------------|-------|---------|
| none | 90.22 ms | 320 MiB | 40 MiB | — |
| TQ 4-bit | 71.49 ms (**−20.8%**) | 64 MiB | 8 MiB | 0.180 |
| TQ 2-bit | 64.82 ms (**−28.1%**) | 32 MiB | 4 MiB | 0.960 |

These numbers include both optimizations applied:
1. Removed tail synchronization from TQ-only path: −1.07 ms (TQ 4-bit)
2. Fused decode-add kernel: −0.68 ms (TQ 4-bit), −1.47 ms (TQ 2-bit)

---

## GPU Scale Limit

```
Job 927250: requested 4 nodes × 8 GPUs = 32 GPUs
Status: PENDING — QOSMaxNodePerJobLimit
```

Current maximum: **2 nodes = 16 GPUs** under `nycugpu_queue / contest_v100` QoS.

---

## Next Steps / What To Try

1. **Report large run (m=65536) as headline** — bigger matrix means larger absolute
   MPI B payload (40 MiB) so compression has more impact.
2. **If QoS opens 4+ nodes**: re-run the 32-GPU probe and see how speedup scales.
3. **TQ 8-bit**: currently swept in v2 but not prominently in v3. Adds a data point
   to the compression-vs-accuracy curve (between none and TQ 4-bit).
4. **Profile MPI latency**: use `mpi_B_gather_decode_time_ms` from the output to
   understand how much of the speedup comes from reduced network time vs CPU
   overhead savings.
5. **Tree all-reduce**: current implementation uses root gather/decode (rank 0
   receives from all ranks). A butterfly/tree reduction would reduce per-rank
   latency at large rank counts.
6. **Smoke test before each major job**: always run the smoke script first to
   confirm the binary and MPI setup work correctly before submitting long sweeps.

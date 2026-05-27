# AI Takeaway — randomized_svd_baseline_v3/

## Purpose

**Final multi-node benchmark**. This is the MPI multi-node extension of v2.
It is the primary location for claiming cross-node speedup from compressed
MPI collectives.

Key addition over v2: the cross-node `B` reduction uses **compressed gather/decode**
(not FP32 MPI_Reduce) when `--compress-b-mode tq/lowbit/tq-qjl/fp16` is set.
This is where the ~20% speedup claim comes from.

Current limit: **16 GPUs** (2 nodes × 8 GPUs) due to Taiwania 2 QoS policy.

---

## Files

| File | Description |
|------|-------------|
| `randomized_svd_multigpu_v3.cu` | Main source. MPI + CUDA. One rank per node, all local GPUs per rank. Includes `fp16` mode (Loop 4). |
| `Makefile` | Requires MPI modules. `module load ucx/1.14.1 openmpi/5.0.2_ucx1.14.1_cuda12.3`. |
| `README.md` | Current results, communication model, build/run instructions. |
| `run_randomized_svd_multigpu_v3_tq_bit_curve_16gpu.slurm` | Main 16-GPU sweep (m=32768). |
| `run_randomized_svd_multigpu_v3_tq_bit_curve_16gpu_large.slurm` | Large 16-GPU sweep (m=65536). |
| `run_randomized_svd_multigpu_v3_smoke_16gpu.slurm` | Quick smoke test: one mode, few repeats. |
| `run_multinode_gpu_visibility_probe_32gpu.slurm` | 32-GPU probe (pending QoS approval). |
| `run_v3_qjl_postfix_verify.slurm` | Loop 4 A1: QJL pre-alloc fix verification on 16-GPU (job 931560). |
| `run_v3_fp16_baseline.slurm` | Loop 4 A2: FP16 MPI gather baseline — none/fp16/TQ-4bit/TQ-2bit at two matrix sizes (job 931564). |

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
| `fp16` | `fp16_gather_decode` | Half-precision payload crosses (2×); near-lossless |

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

### Loop 4 definitive runs (m=32768, n=8192, k=256, 16 GPUs, repeat=100, jobs 931560+931564)

| Mode | Warm Pipeline | MPI B | B Error | vs none |
|------|--------------|-------|---------|---------|
| none | 66.49 ms | 20 MiB | — | — |
| **FP16** | **63.49 ms** | **10 MiB** | **~0.0002** | **−4.5%** |
| **TQ 4-bit** | **49.09 ms** | **4 MiB** | **0.175** | **−26.2%** |
| TQ 2-bit | 46.95 ms | 2 MiB | 0.953 | −29.4% |
| TQ+QJL 4-bit (post-fix) | 82.69 ms | 4 MiB | — | −24.4%† |

†QJL still slower than `none` even post-fix; confirmed dead end.

### Loop 4 definitive runs (m=65536, n=16384, k=256, 16 GPUs, repeat=100, jobs 931560+931564)

| Mode | Warm Pipeline | MPI B | B Error | vs none |
|------|--------------|-------|---------|---------|
| none | 89.72 ms | 40 MiB | — | — |
| **FP16** | **90.24 ms** | **20 MiB** | **~0.0002** | **~0%** |
| **TQ 4-bit** | **71.52 ms** | **8 MiB** | **0.180** | **−20.3%** |
| TQ 2-bit | 64.20 ms | 4 MiB | 0.960 | −28.5% |
| TQ+QJL 4-bit (post-fix) | 133.56 ms | 8 MiB | — | −48.9%† |

†TQ+QJL post-fix: was 379 ms pre-fix, now 133 ms — alloc fix confirmed (−65%).

**FP16 insight**: near-lossless, but 2× MPI compression insufficient vs TQ 4-bit's 5× at
multi-node scale. FP16 pipeline is dominated by the sequential H2D→decode loop on root;
at 64k×16k this fully negates the halved MPI bytes. TQ 4-bit wins decisively.

These numbers include all optimizations (tail sync removal, fused decode-add).

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
   MPI B payload (40 MiB) so compression has more impact. ✓ Done in Loop 4.
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
7. **FP16 optimization (if desired)**: replace the sequential per-rank H2D→decode
   loop with a batch H2D copy of all ranks' data followed by a fused fp16→fp32→add
   kernel. This could recover the FP16 advantage at 64k×16k — but given TQ 4-bit's
   dominance this is low priority.

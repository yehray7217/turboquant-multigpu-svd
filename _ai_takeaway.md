# AI Takeaway — Project Root

## Table of Content

- [AI Takeaway — Project Root](#ai-takeaway--project-root)
  - [Table of Content](#table-of-content)
  - [What Is This Project?](#what-is-this-project)
  - [Directory Map](#directory-map)
  - [What Has Been Done](#what-has-been-done)
    - [Baselines Implemented](#baselines-implemented)
    - [Compression Modes Implemented (in `turboquant/`)](#compression-modes-implemented-in-turboquant)
    - [Key Optimizations Applied](#key-optimizations-applied)
    - [Current Best Results](#current-best-results)
    - [Negative Result](#negative-result)
  - [What Should / Can Be Done Next](#what-should--can-be-done-next)
    - [Highest Priority](#highest-priority)
    - [Experiments To Run / Validate](#experiments-to-run--validate)
  - [Problems — Root-Cause Analysis](#problems--root-cause-analysis)
    - [1. Power Iteration Bug (latent — not yet implemented)](#1-power-iteration-bug-latent--not-yet-implemented)
    - [2. QJL is 3–4x Slower than Pure TQ — Root Cause Found](#2-qjl-is-34x-slower-than-pure-tq--root-cause-found)
    - [3. NCCL (to replace MPI) — Future Work](#3-nccl-to-replace-mpi--future-work)
    - [Profiling / Fine-tuning](#profiling--fine-tuning)
    - [Theory / Algorithm Improvements](#theory--algorithm-improvements)
    - [Documentation](#documentation)
  - [Key Files to Read First](#key-files-to-read-first)
  - [Build Quick Reference](#build-quick-reference)
  - [Run Quick Reference](#run-quick-reference)

## What Is This Project?

Multi-GPU randomized SVD accelerated by **TurboQuant + QJL residual correction**.

- The core idea:
  - when GPUs reduce distributed intermediate matrices ($B_i$ blocks),
    compress them first with low-bit quantization + random rotation (TurboQuant) to
    shrink cross-GPU and cross-node communication volume, then optionally sketch the
    residual with 1-bit QJL signs to partially correct quantization error.

- Platform:
  - **Taiwania 2 HPC cluster**, NVIDIA Tesla V100, CUDA/cuBLAS/cuSOLVER + MPI.

---

## Directory Map

```
turboquant-multigpu-svd/
├── cusolver/               Baseline A: single-GPU exact SVD + MPI multi-GPU proxy app
│   ├── cusolver.cu                 Single-GPU exact SVD (cuSOLVER bidiag-QR)
│   ├── mpi_cusolver.cu             NEW: MPI + CUDA multi-GPU SVD proxy (comm benchmark)
│   └── run_mpi_cusolver.sh         NEW: Slurm script (2 nodes × 8 GPUs)
├── randomized_svd_baseline/   Baseline B v1: partial multi-GPU, compression at Y_i gather
├── randomized_svd_baseline_v2/  v2: TSQR-style, compression at B_i reduction (primary)
├── randomized_svd_baseline_v3/  v3: MPI multi-node compressed collectives (final claim)
├── turboquant/             Shared compression kernel library (TQ + QJL CUDA kernels)
├── slate_baseline/         SLATE library naive randomized SVD (library-level reference)
├── benchmark_compare/      Scripts to compare cuSOLVER vs SLATE vs v2 side-by-side
├── docs/notes/             Theory notes (SVD, Randomized SVD, TurboQuant)
│   └── potential_problem_for_power_iteration.md  NEW: power iteration QR stability fix
├── RECORD.csv              NEW: benchmark result log across all modes and node counts
└── _agent_readme.md        NEW: agent work log (plan / results / conclusions)
```

---

## What Has Been Done

### Baselines Implemented

- **cuSOLVER (Baseline A)**:
  - single-GPU exact SVD via `cusolverDnSgesvd`.
  - Used for
    - correctness verification
    - as library-level timing reference.

- **Randomized SVD v1 (Baseline B)**:
  - multi-GPU pipeline with host-mediated $Y_i$ gather.
  - Compression point is $Y_i$, payload does not grow with GPU count.

- **Randomized SVD v2**:
  - full TSQR-style distributed pipeline.
  - Compression target is $B_i$ reduction (`ngpus * l * n * sizeof(float)`), which grows with GPU count.
  - Single-node, multi-GPU.

- **Randomized SVD v3**:
  - **MPI** multi-node extension of v2.
  - Cross-node $B$ reduction uses compressed gather/decode instead of FP32 MPI_Reduce.
  - Tested up to 16 GPUs (2 nodes × 8 GPUs, Slurm QoS limit).

- **SLATE naive baseline**:
  - Unoptimized SLATE R(randomized)SVD with `Lookahead=0`, `nb=256`.
  - Serves as library distributed randomized SVD reference.

### Compression Modes Implemented (in `turboquant/`)

| Mode     | Description                                                       |
| -------- | ----------------------------------------------------------------- |
| `none`   | FP32 baseline, no compression                                     |
| `lowbit` | Per-block symmetric scalar quantization (8/4/2-bit)               |
| `tq`     | Random Rademacher signs + FWHT rotation + low-bit quant + inverse |
| `tq-qjl` | TQ + QJL residual 1-bit sign sketch correction (exploratory)      |
| `fp16`   | Pure FP16 cast on B_i before P2P (2× compression, ~0.0002 rel-err) — v2 only |

### Key Optimizations Applied

1. `--device-random-input`:
   - generates $A_i$ and $\Omega$ directly on GPU, eliminating host-setup noise from timing.
2. **Repeat timing with warm averages** (`--repeat N`):
   - excludes cold CUDA context initialization from reported performance.
3. **Skip U formation** (`--skip-form-u`) and reconstruction errors from timing hot path.
4. **Removed tail sync** from TQ-only path:
   - `-1.07 ms` on 16-GPU large run.
5. **Fused decode-add kernel**:
   - shared-memory dequant + inverse FWHT + accumulate in one kernel.
   - `-0.68 ms` (4-bit) / `-1.47 ms` (2-bit) on 16 GPUs.
6. **Gaussian QJL samples**:
   - replaced hash/Rademacher signs with Box-Muller Gaussian for correctness.
   - QJL still does not improve $B$ error.

### Current Best Results

**v3 TQ 4-bit on 16 GPUs (m=65536, n=16384, k=256):**

- MPI B payload: 40 MiB → 8 MiB (4.99999× compression)
- Pipeline: 90.22 ms (none) → 71.49 ms (TQ 4-bit) — **~21% speedup**
- B relative error: 0.180

**v3 TQ 2-bit on 16 GPUs (aggressive):**

- MPI B payload: 40 MiB → 4 MiB (~10× compression)
- Pipeline: 90.22 ms → 64.82 ms — **~28% speedup**
- B relative error: 0.960 (too high for accurate SVD, use as compression bound)

**v2 TQ 4-bit on 2 GPUs (single-node reference):**

- B payload: 20 MiB → 4 MiB (4.99998× compression)
- Pipeline: 50.22 ms → 50.11 ms (overhead ≈ negligible at this scale)
- B relative error: 0.171

**v2 1-node 8-GPU mode comparison (m=32768, n=8192, k=256, l=320, job 931176):**

| Mode | Compression | B rel-err | Warm pipeline | `build_reduce_Bi` |
|------|-------------|-----------|---------------|-------------------|
| none | 1× | 0.000 | 45.38 ms | 9.96 ms |
| **FP16** | **2×** | **0.000208** | **43.40 ms** | **6.78 ms** |
| TQ 4-bit | 5× | 0.172 | 42.79 ms | 7.27 ms |
| TQ 2-bit | 10× | 0.948 | 42.56 ms | 6.55 ms( fastest) |
| TQ+QJL 4-bit | 5× | 0.468 | 68.39 ms | 32.75 ms |

Singular value fidelity vs `none` (1.20972, ...):
FP16 = identical, TQ 4-bit +0.7% bias, TQ 2-bit broken (0.350), TQ+QJL +5.4% bias.

---

### Negative Result

**TQ + QJL**:

- Any nonzero `qjl_alpha` worsens B error.
- Best result is always `alpha=0` (equivalent to plain TQ).
- Even with `qjl_dim = d` (full dimension) and Gaussian samples, QJL residual reconstruction does not reduce error.
- The one-bit sketch estimator is too crude for this use case.

---

## What Should / Can Be Done Next

### Highest Priority

1. **Report v3 TQ 4-bit as headline result** (16 GPUs, 21% speedup, 5× compression).
2. **Present v3 TQ 2-bit as aggressive compression point** with explicit caveat on B error (0.960) — show the compression vs accuracy tradeoff curve.
3. **Keep QJL as documented negative result** — motivate future redesign.

### Experiments To Run / Validate

4. **v2 8-GPU single-node**: run `run_randomized_svd_multigpu_v2_tq_bit_curve_8gpu.slurm` to show how compression benefit scales with GPU count on one node.
5. **[Full reconstruction error vs. bits] curve**:
   - compare `||A - U Σ Vᵀ||_F / ||A||_F` across none/TQ-4bit/TQ-2bit at multiple matrix sizes.

## Problems — Root-Cause Analysis

### 1. Power Iteration Bug (latent — not yet implemented)

The naive pseudocode `Y = A(AᵀY)` repeated q times is numerically unstable.
All columns of Y collapse toward the dominant singular vector after a few steps.
**Fix**: interleave QR after every matrix multiply (subspace iteration).
In multi-GPU: each QR in the loop requires a distributed TSQR → adds communication.
Reference: `docs/notes/potential_problem_for_power_iteration.md`.

### 2. QJL is 3–4x Slower than Pure TQ — Root Cause Found

From `RECORD.csv`:

- TQ 4-bit (1 node, 8 GPU): **43.1 ms**
- TQ+QJL 4-bit (same): **161.9 ms** → +118.8 ms overhead!

**Root cause**: `quantize_fp32_device_column_tq_to_device_payload` allocates THREE
GPU scratch buffers per call (`d_reconstructed`, `d_residual`, `d_qjl_partials`),
and frees them at end. With `cudaMalloc` taking ~1-2ms for 10MB+ allocations,
and 8 GPUs processed sequentially in the hot loop → 8 × 3 allocs × 2ms ≈ 48ms.
Add the corresponding frees and a forced `cudaStreamSynchronize` for residual norm
→ total ~120ms overhead per iteration. Matches observed data.

**Fix**: pre-allocate these scratch buffers in setup (alongside the existing
`d_Bi_tq_work`), pass them as optional external pointers to the compress function.
This is implemented in `_agent_readme.md` Loop 1.

### 3. NCCL (to replace MPI) — Future Work

Current v3 uses `MPI_Gather`/`MPI_Allgather` for cross-node B reduction.
NCCL (NVIDIA Collective Communications Library) is GPU-native and can be faster
on NVLink-equipped nodes. Taiwania 2 uses InfiniBand, so the gain may be smaller.
Requires significant rewrite. Lower priority than fixing QJL.

### Profiling / Fine-tuning

6. **Nsight Compute on TQ FWHT kernels**:
   - the FWHT pack/unpack kernels are next bottleneck candidates after fused decode-add. Profile `compress_Bi` kernel time.
7. **Per-column vs per-block scaling**: current lowbit uses a single global scale per $B_i$ block. Per-column scales might reduce quantization error at same bits.
8. **Power iteration**: add `q` rounds of `Y = A(AᵀY)` in the randomized SVD
   projection step to improve singular value capture quality.

### Theory / Algorithm Improvements

9. **QJL redesign**:
   - instead of sketching the residual after TQ, integrate QJL directly into the inner-product estimator for `B_i = Qᵢᵀ Aᵢ`.
   - This is the approach suggested in the paper but not yet implemented.
10. **~~FP16 communication baseline~~ — DONE (job 931176)**:
    - `--compress-b-mode fp16` added to v2: 2× compression, 0.0002 B-error, 43.4 ms warm pipeline.
    - On the reduce step alone FP16 (6.78 ms) is actually faster than TQ 4-bit (7.27 ms),
      but TQ 4-bit's 5× compression wins for multi-node where bandwidth dominates.
    - Confirms TQ 4-bit is the right headline result. v3 multi-node FP16 port still TODO.
11. **Tree all-reduce for MPI B**:
    - current v3 uses root gather/decode. A tree
      reduction would reduce communication latency at large node counts.

### Documentation

12. **Complete `docs/notes/TurboQuant.md`**: currently marked "施工中" (under construction).
13. **Add experiment results tables** to the root README after final runs.

---

## Key Files to Read First

| File                                   | Why                                                    |
| -------------------------------------- | ------------------------------------------------------ |
| `OPTIMIZATION_HISTORY.md`              | Complete experiment log with all results and decisions |
| `randomized_svd_baseline_v3/README.md` | Current final claims and result tables                 |
| `turboquant/turboquant.hpp`            | Public API for all compression modes                   |
| `randomized_svd_baseline_v2/README.md` | Compression options, timing policy, profiling guide    |
| `docs/notes/Randomized_SVD.md`         | Algorithm theory with pseudocode                       |

---

## Build Quick Reference

```bash
# v2 single-node
module load cuda/12.8
cd randomized_svd_baseline_v2 && make

# v3 multi-node
module load cuda/12.8 ucx/1.14.1 openmpi/5.0.2_ucx1.14.1_cuda12.3
cd randomized_svd_baseline_v3 && make

# cuSOLVER baseline
cd cusolver && make
```

## Run Quick Reference

```bash
# Main v3 comparison (16 GPU)
sbatch randomized_svd_baseline_v3/run_randomized_svd_multigpu_v3_tq_bit_curve_16gpu_large.slurm

# v2 large 2-GPU TQ sweep
sbatch randomized_svd_baseline_v2/run_randomized_svd_multigpu_v2_tq_bit_curve.slurm

# Side-by-side benchmark comparison
sbatch benchmark_compare/run_compare_svd_2gpu.slurm
```

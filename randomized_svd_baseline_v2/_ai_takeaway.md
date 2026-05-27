# AI Takeaway — randomized_svd_baseline_v2/

## Purpose

**Primary single-node benchmark**. This is the TSQR-style multi-GPU randomized
SVD with TurboQuant/QJL `B_i` compression. This directory is the main place
for:
- Single-node kernel profiling (Nsight Systems / Nsight Compute)
- Tuning and debugging compression kernels
- Producing bit-vs-error curves on 2 GPUs

v3 is the multi-node MPI version of this same pipeline.

---

## Files

| File | Description |
|------|-------------|
| `randomized_svd_multigpu_v2.cu` | Full TSQR pipeline + compression. Main source. Includes `fp16` mode (local; no turboquant dep). |
| `Makefile` | Links against `../turboquant/turboquant.cu`. `nvcc -O3 -std=c++17 -arch=sm_70`. |
| `README.md` | Comprehensive: compression options, timing policy, profiling guide. |
| `OPTIMIZATION_HISTORY.md` | Copy of root OPTIMIZATION_HISTORY.md — same content. |
| `run_randomized_svd_multigpu_v2.slurm` | Main sweep (all modes: none/lowbit/tq/tq-qjl at 8/4/2-bit). |
| `run_randomized_svd_multigpu_v2_large.slurm` | Large 2-GPU: m=32768, n=8192, k=256. |
| `run_randomized_svd_multigpu_v2_tq_bit_curve.slurm` | Bit-vs-error curve sweep. |
| `run_randomized_svd_multigpu_v2_tq_focus.slurm` | Focused TQ/TQ+QJL comparison. |
| `run_randomized_svd_multigpu_v2_qjl_alpha_grid.slurm` | Grid search over qjl_alpha. |
| `run_randomized_svd_multigpu_v2_qjl_dim_sweep.slurm` | Grid search over qjl_dim. |
| `run_randomized_svd_multigpu_v2_qjl_dim_d.slurm` | QJL with dim=l (full vector dimension). |
| `run_randomized_svd_multigpu_v2_nsys.slurm` | Nsight Systems timeline profiling. |
| `run_randomized_svd_multigpu_v2_ncu.slurm` | Nsight Compute kernel profiling. |
| `run_randomized_svd_multigpu_v2_scaling_1_2_4_8.slurm` | GPU count scaling study. |
| `run_randomized_svd_multigpu_v2_device_random.slurm` | CUDA-focused: device-side input gen. |
| `run_randomized_svd_multigpu_v2_accuracy.slurm` | Accuracy check (no skip-form-u). |
| `run_randomized_svd_multigpu_v2_qjl_*.slurm` | Various QJL parameter explorations. |
| `run_randomized_svd_multigpu_v2_tq_bit_curve_8gpu.slurm` | TQ curve on 8 GPUs. |
| `run_multinode_gpu_visibility_probe.slurm` | Multi-node GPU visibility diagnostic. |
| `run_qjl_fix_benchmark.slurm` | 1-node 8-GPU timing benchmark — used to verify the QJL pre-allocation fix (job 928869). |
| `run_qjl_berror_postfix.slurm` | 1-node 8-GPU B-error check for QJL post-fix (job 928995 confirmed QJL is a dead end). |
| `run_fp16_baseline.slurm` | NEW: 1-node 8-GPU benchmark adding `--compress-b-mode fp16` to the comparison (job 931176). |

---

## Key Concepts

### Pipeline Flow

```
GPU i: A_i (row block)
GPU i: Y_i = A_i * Omega        [cuBLAS dgemm]
GPU i: Y_i = Qbar_i * R_i       [cuSOLVER geqrf]
GPU 0: TSQR reduce R_i           [gather small R factors, host-side TSQR]
GPU i: Q_i = Qbar_i * T_i        [apply TSQR transform]
GPU i: B_i = Q_i^T * A_i         [cuBLAS dgemm]
GPU i: compress B_i               [turboquant kernels]  ← COMPRESSION POINT
GPU 0: B = sum_i B_i_hat          [accumulate on GPU 0]
GPU 0: SVD(B)                     [cuSOLVER gesvd]
GPU i: U_i = Q_i * U_tilde        [cuBLAS dgemm, optional via --skip-form-u]
```

### Why B_i Is the Right Compression Target

`B_i` payload = `ngpus * l * n * sizeof(float)` — grows with GPU count.
For m=32768, n=8192, k=256, l=320, ngpus=2: **20 MiB** FP32 total.
TQ 4-bit compresses this to ~4 MiB (4.99998× ratio).

Compare to v1's Y_i gather: always `m * l * sizeof(float)` = constant ~40 MiB,
not scaling with GPU count (compressing it helps less as you add GPUs).

### Timing Policy (Important)

Always use **warm averages** (`warm_pipeline_avg_ms`):
- Use `--repeat 100` (or larger)
- The first iteration is reported separately as `cold_pipeline_ms`
- Cold start includes CUDA context init, cuBLAS handle creation, cuSOLVER workspace setup

Always use **device-side random input** (`--device-random-input`) for timing:
- Eliminates host matrix allocation and H→D copy from timing

Always use **skip-form-u** (`--skip-form-u`) for performance runs:
- U formation is not part of the compression story

---

## How to Build & Run

```bash
module load cuda/12.8
cd randomized_svd_baseline_v2
make

# Standard large 2-GPU sweep
sbatch run_randomized_svd_multigpu_v2_tq_bit_curve.slurm

# Profiling
sbatch run_randomized_svd_multigpu_v2_nsys.slurm  # timeline
sbatch run_randomized_svd_multigpu_v2_ncu.slurm   # kernel bottlenecks
```

---

## Key CLI Options

| Option | Meaning |
|--------|---------|
| `--m`, `--n`, `--k`, `--oversample` | Matrix shape and rank |
| `--ngpus <g>` | Number of GPUs |
| `--compress-b-mode none\|lowbit\|tq\|tq-qjl\|fp16` | Compression mode. `fp16` ignores `--compress-b-bits` (must be 0). |
| `--compress-b-bits 8\|4\|2` | Bit width |
| `--qjl-dim <d>` | QJL sketch dimension |
| `--qjl-alpha <a>` | QJL correction strength (0.0 = TQ only) |
| `--device-random-input` | Generate A_i on GPU (use for timing) |
| `--skip-form-u` | Skip final U_i = Q_i U_tilde step |
| `--no-check-error` | Skip full A reconstruction error |
| `--no-check-b-error` | Skip B_i compression error check |
| `--repeat N` | Run N times, report warm average |
| `--repeat-print-every K` | Print progress every K repeats |

---

## Key Reported Metrics

```
warm_pipeline_avg_ms       Main headline: steady-state wall time
reduce_B_compression_ratio Achieved compression ratio
reduce_B_payload_MiB       Actual bytes sent
reduce_B_relative_error    ||B_i_hat - B_i||_F / ||B_i||_F  (compression quality)
compress_Bi_time_ms        Time spent in compression kernels only
```

---

## Reference Results (2-GPU, m=32768, n=8192, k=256, l=320)

| Mode | Payload | Ratio | Warm Pipeline | B Error |
|------|---------|-------|---------------|---------|
| none | 20 MiB | 1.00× | 50.22 ms | — |
| TQ 4-bit | 4.00 MiB | 4.99998× | 50.11 ms | 0.171 |
| TQ 2-bit | 2.00 MiB | 9.99992× | 49.74 ms | 0.949 |
| TQ+QJL 4-bit | 4.00 MiB | ~5× | 52.95 ms | worse than TQ |

## Reference Results (8-GPU, m=32768, n=8192, k=256, l=320, job 931176)

| Mode | Payload | Ratio | Warm Pipeline | `build_reduce_Bi` | B Error |
|------|---------|-------|---------------|-------------------|---------|
| none | 80 MiB | 1.00× | 45.38 ms | 9.96 ms | 0.000 |
| **fp16** | **40 MiB** | **2.00×** | **43.40 ms** | **6.78 ms** | **0.000208** |
| TQ 4-bit | 16.0 MiB | 4.99998× | 42.79 ms | 7.27 ms | 0.172 |
| TQ 2-bit | 8.00 MiB | 9.99992× | 42.56 ms | 6.55 ms | 0.948 |
| TQ+QJL 4-bit | 16.0 MiB | 4.99937× | 68.39 ms | 32.75 ms | 0.468 |

Notes:
- FP16 has the fastest `build_reduce_Bi` (P2P bytes halved) and singular values
  identical to `none` (1.20972, ...) to 5 decimal places.
- TQ 4-bit's singular values drift slightly (1.21848, +0.7% bias) — accuracy cost of compression.
- Pipeline differences are small (42–45 ms) because at this matrix size the bottleneck is
  SVD on GPU 0 (18 ms) + local QR (8–9 ms), not the reduce step. TQ's compression advantage
  matters more at multi-node scale (v3).

---

## Nsight Profiling Summary

From a single-iteration TQ 4-bit profile (m=32768, n=8192, k=256, ngpus=2):

| Region | Time |
|--------|------|
| local_projection_Yi | 255 ms (dominates) |
| svd_B_on_gpu0 | 71 ms |
| local_qr_Yi | 38 ms |
| build_reduce_Bi | 20 ms |
| compress_Bi | 9.5 ms |
| tsqr_R_reduce_gpu0 | 5.5 ms |

**Conclusion**: compression (`compress_Bi`) is a small fraction of the full
pipeline in single-node 2-GPU mode. At 16 GPUs (v3), the MPI B transfer
becomes dominant and compression saves 19–22%.

---

## Next Steps / What To Try

1. Run `run_randomized_svd_multigpu_v2_tq_bit_curve_8gpu.slurm` to see how
   compression benefit grows with GPU count on one node.
2. Use `run_randomized_svd_multigpu_v2_ncu.slurm` to profile TQ FWHT and
   pack/unpack kernels — next optimization candidates.
3. Try per-column scaling in the lowbit path to reduce quantization error at
   the same bit width.
4. For accuracy analysis (not timing), use `run_randomized_svd_multigpu_v2_accuracy.slurm`
   which re-enables U formation and reconstruction error.

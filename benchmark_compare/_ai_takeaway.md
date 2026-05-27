# AI Takeaway — benchmark_compare/

## Purpose

**Side-by-side comparison scripts** for presenting all three SVD implementations
together in one Slurm job:

1. cuSOLVER (exact SVD, single GPU)
2. SLATE RSVD (library distributed randomized SVD, 2 GPUs)
3. v2 RSVD with TurboQuant options (custom distributed randomized SVD, 2 GPUs)

This is the **presentation layer** — it does not contain source code, only
the orchestration scripts that produce the headline comparison table.

---

## Files

| File                         | Description                                                      |
| ---------------------------- | ---------------------------------------------------------------- |
| `run_compare_svd_2gpu.slurm` | Main comparison script. Runs all three methods in sequence.      |
| `README.md`                  | Setup instructions, what metrics to compare, presentation story. |

---

## Key Concepts

### What Each Method Measures

| Method     | Algorithm                     | GPU count       | Metric                           |
| ---------- | ----------------------------- | --------------- | -------------------------------- |
| cuSOLVER   | Exact SVD (full dense)        | 1               | `GPU SVD Execution Time`         |
| SLATE RSVD | Library randomized rank-k SVD | 2 (2 MPI ranks) | `total RSVD core`, per-stage     |
| v2 RSVD    | Custom randomized rank-k SVD  | 2               | `warm_compute_avg_ms`, per-stage |

**Important**: cuSOLVER runs a **square matrix** (4096×4096, 8192×8192) because
it computes full SVD. SLATE and v2 use the rectangular benchmark shape
(m=32768, n=8192) to represent a realistic workload. They are **not directly
comparable** in absolute time — cuSOLVER is a different algorithm.

The fair algorithmic comparison is **SLATE vs v2 none** (both are randomized,
both 2-GPU). Then **v2 none vs v2 TQ/TQ-QJL** shows the compression benefit.

### Comparison Script Runs

```text
cuSOLVER:  M=N=4096 and M=N=8192  (square, single GPU, exact SVD)
SLATE:     M=32768, N=8192, K=256, oversample=64, 2 ranks / 2 GPUs
v2:        M=32768, N=8192, K=256, oversample=64, 2 GPUs
           modes: none, tq 4-bit, tq-qjl 4-bit
           --repeat 1000  (reports warm timing)
```

---

## How to Run

### Prerequisites

SLATE must be installed (one-time setup):

```bash
cd slate_baseline
SLATE_BUILD_JOBS=2 ./start_background_setup.sh
./check_background_setup.sh  # wait until done
```

### Run comparison

```bash
cd /path/to/turboquant-multigpu-svd
sbatch benchmark_compare/run_compare_svd_2gpu.slurm
```

---

## Presentation Story

```
Step 1: cuSOLVER
  → Exact SVD, single GPU, library-level reference
  → Shows baseline cost of exact decomposition

Step 2: SLATE RSVD (none)
  → Randomized SVD, distributed, naive library baseline
  → Shows cost of distributed approximate SVD without compression

Step 3: v2 RSVD (none)
  → Our custom TSQR pipeline, same algorithm class as SLATE
  → Should be comparable to SLATE (validates our implementation)

Step 4: v2 RSVD (TQ 4-bit)
  → Same pipeline, but B_i reduction uses TurboQuant compression
  → Shows compression speedup relative to step 3

Step 5: v2 RSVD (TQ+QJL 4-bit)
  → Adds QJL residual correction (exploratory)
  → Currently does not improve over TQ alone
```

---

## Key Metrics to Extract

```
cuSOLVER output:
  "GPU SVD Execution Time: X ms"

SLATE output:
  "total RSVD core: X ms"
  "Y = A*Omega: X ms"
  "geqrf(Y): X ms"
  "B = Q^T*A: X ms"
  "svd(B): X ms"

v2 output (use Repeat summary block):
  warm_compute_avg_ms    ← HEADLINE for v2
  warm_compute_min_ms
  local_projection_Yi
  local_qr_Yi
  build_reduce_Bi
  svd_B_on_gpu0
  reduce_B_payload_MiB
  reduce_B_compression_ratio
  reduce_B_relative_error
```

**Do not use `cold_pipeline_ms` for presentation** — always use the warm average.

---

## Next Steps / What To Try

1. **Run this script** after any significant change to v2 to get an updated
   three-way comparison in one job.
2. **Add v3 to the comparison**: currently this script only covers single-node v2.
   For the multi-node story, run `randomized_svd_baseline_v3/run_randomized_svd_multigpu_v3_tq_bit_curve_16gpu_large.slurm`
   separately and present alongside this table.
3. **Add lowbit modes to comparison**: currently only `none`, `tq`, `tq-qjl` are
   compared. Adding `lowbit 8-bit` / `lowbit 4-bit` gives a richer tradeoff curve.
4. **If SLATE is not installed**: this is the most common setup blocker. Verify
   with `cd slate_baseline && source env_taiwania2.sh && make doctor`.

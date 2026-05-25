# SVD Method Comparison

This folder contains scripts for comparing:

```text
cuSOLVER exact SVD
SLATE randomized SVD
v2 randomized SVD with optional TurboQuant/QJL
```

The comparison has two separate meanings:

```text
cuSOLVER  exact/full dense SVD on one GPU
SLATE     randomized rank-k SVD on two GPUs through SLATE/MPI
v2        randomized rank-k SVD on two GPUs with optional B_i compression
```

So cuSOLVER is the exact-SVD reference, while SLATE and v2 are the fairer
algorithmic comparison for randomized SVD.

## Run

If SLATE is not installed yet, set it up once:

```bash
cd /home/yehray7217/turboquant-multigpu-svd/slate_baseline
SLATE_BUILD_JOBS=2 ./start_background_setup.sh
./check_background_setup.sh
```

Wait until `check_background_setup.sh` reports that no background process is
running and the log ends with `background setup done`.

Then run the comparison:

```bash
cd /home/yehray7217/turboquant-multigpu-svd
sbatch benchmark_compare/run_compare_svd_2gpu.slurm
```

The script runs:

```text
cuSOLVER:
  M=N=4096
  M=N=8192

SLATE RSVD:
  M=32768, N=8192, K=256, oversample=64, 2 MPI ranks / 2 GPUs

v2 RSVD:
  M=32768, N=8192, K=256, oversample=64, 2 GPUs
  modes: none, tq 4-bit, tq-qjl 4-bit
  uses --repeat 1000 and reports warm timing after excluding the first repeat
```

The cuSOLVER case is square because the current cuSOLVER baseline computes
full singular vectors and is not the same algorithm as randomized SVD. The
SLATE and v2 cases use the same rectangular benchmark shape as the large v2
sweep.

## What To Compare

Use these fields:

```text
cuSOLVER:
  GPU SVD Execution Time

SLATE:
  total RSVD core
  Y = A*Omega
  geqrf(Y)
  B = Q^T*A
  svd(B)

v2:
  warm_compute_avg_ms
  warm_compute_min_ms
  cold_pipeline_ms
  local_projection_Yi
  local_qr_Yi
  build_reduce_Bi
  svd_B_on_gpu0
  reduce_B_payload_MiB
  reduce_B_relative_error
  Fast relative Frobenius reconstruction error
```

For presentation, the clean story is:

```text
cuSOLVER: exact but single-GPU/full-SVD reference
SLATE: library distributed randomized SVD baseline
v2 none: our distributed randomized SVD baseline
v2 TQ/TQ+QJL: our communication-compressed variants
```

For v2, use the `Repeat summary` block for headline performance. The first
repeat includes CUDA/cuBLAS/cuSOLVER lazy initialization effects, so the
presentation number should be `warm_compute_avg_ms` or `warm_compute_min_ms`,
not `cold_pipeline_ms`.

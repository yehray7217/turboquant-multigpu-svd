# TQ4 vs NONE N-Grid Timing

This experiment measures how v4 timing changes as the column dimension `n`
grows from `16k` to `320k` in a `16k` grid.

It compares only:

```text
NONE
TQ4
```

QJL is intentionally excluded.

## Default Configuration

| Parameter | Value |
|---|---:|
| `m` | 65536 |
| `n` | 16384, 32768, ..., 327680 |
| `k` | 250 |
| `oversample` | 6 |
| `l` | 256 |
| GPUs | 8 V100 on 1 node |
| `subspace_iter` | 1 |
| input | device-random synthetic matrix |
| spectrum | polynomial, param `0.6`, rank `8192` |
| repeat | 5 |
| final error | skipped |

The run skips final reconstruction error because this is a timing curve across
20 matrix sizes and 2 methods. The goal is to see where TQ4 starts to pull away
from NONE as `n` grows.

## Run

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_tq4_none_n_grid
sbatch run_tq4_none_n_grid_8gpu.slurm
```

To change repeat count or range:

```bash
REPEAT=10 N_START=16384 N_STOP=327680 N_STEP=16384 sbatch run_tq4_none_n_grid_8gpu.slurm
```

## Summarize

After the job finishes:

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_tq4_none_n_grid
python3 summarize_n_grid.py tq4_none_n_grid_8gpu_<jobid>.out > tq4_none_n_grid_8gpu_<jobid>.csv
cat tq4_none_n_grid_8gpu_<jobid>.csv
```

The CSV columns are:

```text
method,matrix_size,m,n,gpu,repeat,total_ms,speedup_vs_none,host_gpu_mib,nvlink_mib,ib_mib
```

## Notes

This is a scaling/timing experiment, not an accuracy experiment. Use the fixed
`64k x 16k` validation runs in `../exp_cuda_opt_round1` for B error and final
reconstruction error after CUDA kernel changes.

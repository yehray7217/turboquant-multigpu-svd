# TQ vs NONE N-Grid Timing/Error

This experiment measures how v4 timing changes as the column dimension `n`
grows from `16k` to `320k` in a `16k` grid.

It compares:

```text
NONE
TQ4
TQ2
TQ1
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
| final reconstruction error | skipped |
| B compression error | enabled by default |

The run skips final reconstruction error because this is a timing curve across
20 matrix sizes and 4 methods. It enables `--check-b-error` by default so TQ1
and TQ2 report their local/global `B` relative error while keeping the timing
summary available.

## Run

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_tq4_none_n_grid
sbatch run_tq4_none_n_grid_8gpu.slurm
```

To change repeat count or range:

```bash
REPEAT=10 N_START=16384 N_STOP=327680 N_STEP=16384 sbatch run_tq4_none_n_grid_8gpu.slurm
```

To disable B-error checks for a pure timing pass:

```bash
CHECK_B_ERROR=0 sbatch run_tq4_none_n_grid_8gpu.slurm
```

## Summarize

After the job finishes:

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_tq4_none_n_grid
python3 summarize_n_grid.py tq_n_grid_8gpu_<jobid>.out > tq_n_grid_8gpu_<jobid>.csv
cat tq_n_grid_8gpu_<jobid>.csv
```

The CSV columns are:

```text
method,matrix_size,m,n,gpu,repeat,total_ms,speedup_vs_none,host_gpu_mib,nvlink_mib,ib_mib,global_b_rel_error_pct,local_bi_rel_error_pct
```

## Notes

This is still primarily a scaling/timing experiment. The error columns are B
compression error, not final reconstruction error. Use a smaller fixed-size
validation run with final error enabled if you need full end-to-end accuracy.

# 1-Node 1/2/4-GPU NONE vs TQ4 N-Grid

This experiment compares NONE and TQ4 while varying both matrix width and the
number of GPUs used on one node.

```text
GPU counts: 1, 2, 4
n grid:     16k, 32k, ..., 320k
```

The Slurm job allocates four GPUs. Each program invocation uses only the first
`1`, `2`, or `4` visible GPUs through `--gpus-per-rank`.

## Run

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_tq4_none_n_grid_1node_scaling
sbatch run_tq4_none_n_grid_1node_scaling.slurm
```

Run a short smoke test first:

```bash
N_STOP=32768 REPEAT=2 sbatch run_tq4_none_n_grid_1node_scaling.slurm
```

Change the tested GPU counts or enable final reconstruction error:

```bash
GPU_COUNTS="1 2 4" ERROR_MODE=final sbatch run_tq4_none_n_grid_1node_scaling.slurm
```

## Summarize

```bash
python3 summarize_n_grid_1node_scaling.py \
  tq4_none_n_grid_1node_scaling_<jobid>.out \
  > tq4_none_n_grid_1node_scaling_<jobid>.csv
```

Important columns:

- `speedup_vs_none`: NONE time divided by TQ4 time at the same `n` and GPU count.
- `scaling_vs_1gpu`: one-GPU time divided by current time for the same method and `n`.
- `host_gpu_mib`, `nvlink_mib`, `ib_mib`: measured communication payloads.

The default run is timing-only. Set `ERROR_MODE=final` to also report
end-to-end final reconstruction error.

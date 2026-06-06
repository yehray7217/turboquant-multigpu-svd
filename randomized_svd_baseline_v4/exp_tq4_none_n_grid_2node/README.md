# 2-Node NONE vs TQ4 N-Grid

This experiment measures the current v4 NONE and TQ4 paths on:

```text
2 nodes x 8 V100 GPUs = 16 GPUs
2 MPI ranks, one rank per node
```

The default grid holds `m=65536` constant and increases `n` from `16k` to
`320k` in `16k` steps. It measures the full multi-node pipeline, including
node-local peer copies, host staging, and MPI/InfiniBand communication.

## Run

The default is a pure timing run with five repeats:

```bash
cd ~/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_tq4_none_n_grid_2node
sbatch run_tq4_none_n_grid_2node.slurm
```

For a shorter smoke test:

```bash
N_STOP=32768 REPEAT=2 sbatch run_tq4_none_n_grid_2node.slurm
```

To include end-to-end final reconstruction error:

```bash
ERROR_MODE=final sbatch run_tq4_none_n_grid_2node.slurm
```

## Summarize

```bash
python3 summarize_n_grid_2node.py tq4_none_n_grid_2node_<jobid>.out \
  > tq4_none_n_grid_2node_<jobid>.csv
```

The main comparison columns are:

```text
matrix_size,total_ms,speedup_vs_none,host_gpu_mib,nvlink_mib,ib_mib
```

The TQ4 speedup is calculated against NONE at the same matrix size.

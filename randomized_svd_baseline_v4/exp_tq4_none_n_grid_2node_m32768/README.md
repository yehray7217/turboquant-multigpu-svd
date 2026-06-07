# 2-Node NONE vs TQ4 N-Grid (fixed m = 32768) — Communication Payload Scaling

This is a **communication payload scaling** experiment, **not** a balanced
matrix scaling experiment.

We **hold `m = 32768` constant** and **increase only `n`**, from `16k` up to
`640k` in `16k` steps. The point is to isolate how TQ4 helps as the
**communication payload grows**, independently of the local compute shape.

## Why fix m and grow n?

The two communication-heavy reduces in distributed randomized SVD are the
subspace-iteration reduce `Z = sum_i A_i^T Q_i` and the projected-matrix reduce
`B = sum_i Q_i^T A_i`. Both are `n x l` matrices, so their per-step payload is:

```text
O(n * l)
```

with `l = k + oversample = 256` fixed here. Therefore growing `n` (with `m`,
`k`, `l` all fixed) grows the reduce payload **linearly in n**, while keeping
the per-GPU row-block shape and the `O(l^2)` TSQR cost essentially unchanged.
This makes `n` a clean knob for "how big is the communication payload", and lets
us ask: **does the TQ4 speedup over NONE keep rising as the payload grows, or
does it converge?**

```text
2 nodes x 8 V100 GPUs = 16 GPUs
2 MPI ranks, one rank per node
m = 32768 (fixed)
n = 16384, 32768, 49152, ..., 655360   (step 16384)
k = 250, oversample = 6, l = 256
subspace_iter = 1
spectrum = polynomial, sigma_i = i^-0.6, spectrum_rank = 8192
repeat = 5
```

## Methods

| Method | compress-b-mode | compress-b-bits | compress-subspace-mode | compress-subspace-bits |
|---|---|---:|---|---:|
| NONE | none | 0 | none | 0 |
| TQ4  | tq   | 4 | tq   | 4 |

## Run

The default is a **timing-only** run (`ERROR_MODE=none`, `CHECK_B_ERROR=0`) with
five repeats:

```bash
cd /home/terryyang1/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_tq4_none_n_grid_2node_m32768
sbatch run_tq4_none_n_grid_2node_m32768.slurm
```

The full sweep is large (40 `n` values x 2 methods, with the matrix growing up
to `32768 x 655360`). The `nycugpu_queue` wall limit is 1 hour, so you may need
to split the sweep across several jobs using the `N_START` / `N_STOP` env vars:

```bash
N_START=16384  N_STOP=327680 sbatch run_tq4_none_n_grid_2node_m32768.slurm
N_START=344064 N_STOP=655360 sbatch run_tq4_none_n_grid_2node_m32768.slurm
```

Shorter smoke test:

```bash
N_STOP=32768 REPEAT=2 sbatch run_tq4_none_n_grid_2node_m32768.slurm
```

To also measure end-to-end final reconstruction error (slower):

```bash
ERROR_MODE=final sbatch run_tq4_none_n_grid_2node_m32768.slurm
```

## Summarize

```bash
python3 summarize_n_grid_2node_m32768.py tq4_none_n_grid_2node_m32768_<jobid>.out \
  > tq4_none_n_grid_2node_m32768_<jobid>.csv
```

CSV columns:

```text
method,nodes,mpi_ranks,gpu,matrix_size,m,n,repeat,total_ms,
speedup_vs_none,host_gpu_mib,nvlink_mib,ib_mib,final_reconstruction_error
```

`speedup_vs_none = NONE total_ms / TQ4 total_ms` at the same `n`. In timing-only
mode `final_reconstruction_error` is left blank.

## Results

Timing-only by default. Fill in after running (one row per `n`, comparing the
two methods); `final_reconstruction_error` only populated when `ERROR_MODE=final`.

| n | NONE total (ms) | TQ4 total (ms) | Speedup | Host-GPU payload (MiB) | NVLink payload (MiB) | InfiniBand payload (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 16384 | | | | | | |
| 32768 | | | | | | |
| 49152 | | | | | | |
| ... | | | | | | |
| 655360 | | | | | | |

> Payload columns above are for whichever method you tabulate; the CSV reports
> Host-GPU / NVLink / InfiniBand payload per method, so you can compare NONE vs
> TQ4 payload directly and confirm the ~O(n) growth of the compressed reduce.

## Expected story

If TQ4 compresses the `O(nl)` reduce traffic, its speedup over NONE should
**increase with `n`** as communication takes a larger share of total time, then
**converge** once the reduce dominates and the fixed compute/quantization
overheads become negligible. This experiment checks whether the speedup is still
climbing at `n = 640k` or has already flattened.

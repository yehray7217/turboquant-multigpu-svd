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

Timing-only run (`ERROR_MODE=none`, `repeat=5`), Taiwania 2 `nycugpu_queue`,
2 nodes x 8 V100, account ACD115064, job 947279. Source:
`tq4_none_n_grid_2node_m32768_947279.out` -> `.csv` via
`summarize_n_grid_2node_m32768.py`. `Speedup = NONE total_ms / TQ4 total_ms`
(warm means). Payload cells show `NONE -> TQ4` (MiB).

> Hardware cap: the full sweep was capped at **n = 458752**. The B/Z reduce
> gathers all 15 peer `l x n` blocks onto GPU 0, so GPU-0 memory grows ~O(n);
> on the 32 GB V100s every `n >= 475136` (i.e. 475136, 491520, ..., 655360)
> hits CUDA out-of-memory at `randomized_svd_multigpu_v4.cu:1153`. The reachable
> grid is therefore the 28 points `16384 ... 458752`.

| n | NONE total (ms) | TQ4 total (ms) | Speedup | Host-GPU MiB (NONE->TQ4) | NVLink MiB (NONE->TQ4) | InfiniBand MiB (NONE->TQ4) |
|---:|---:|---:|---:|---:|---:|---:|
| 16384 | 174 | 94 | 1.85x | 368->117 | 448->282 | 80->24 |
| 32768 | 359 | 149 | 2.40x | 704->201 | 896->564 | 144->32 |
| 49152 | 512 | 199 | 2.57x | 1040->286 | 1344->845 | 208->41 |
| 65536 | 666 | 254 | 2.62x | 1376->370 | 1792->1127 | 272->49 |
| 81920 | 829 | 311 | 2.66x | 1712->455 | 2240->1409 | 336->57 |
| 98304 | 994 | 362 | 2.75x | 2048->540 | 2688->1690 | 400->66 |
| 114688 | 1150 | 420 | 2.74x | 2384->624 | 3136->1972 | 464->74 |
| 131072 | 1305 | 491 | 2.66x | 2720->709 | 3584->2254 | 528->82 |
| 147456 | 1469 | 543 | 2.70x | 3056->794 | 4032->2536 | 592->90 |
| 163840 | 1638 | 593 | 2.76x | 3392->878 | 4480->2818 | 656->98 |
| 180224 | 1789 | 658 | 2.72x | 3728->963 | 4928->3099 | 720->107 |
| 196608 | 1947 | 688 | 2.83x | 4064->1048 | 5376->3381 | 784->115 |
| 212992 | 2091 | 769 | 2.72x | 4400->1132 | 5824->3663 | 848->123 |
| 229376 | 2252 | 816 | 2.76x | 4736->1217 | 6272->3944 | 912->132 |
| 245760 | 2400 | 887 | 2.71x | 5072->1301 | 6720->4226 | 976->140 |
| 262144 | 2545 | 959 | 2.65x | 5408->1386 | 7168->4508 | 1040->148 |
| 278528 | 2718 | 1017 | 2.67x | 5744->1471 | 7616->4790 | 1104->156 |
| 294912 | 2905 | 1080 | 2.69x | 6080->1555 | 8064->5072 | 1168->164 |
| 311296 | 3018 | 1130 | 2.67x | 6416->1640 | 8512->5353 | 1232->173 |
| 327680 | 3160 | 1195 | 2.64x | 6752->1724 | 8960->5635 | 1296->181 |
| 344064 | 3365 | 1243 | 2.71x | 7088->1809 | 9408->5917 | 1360->189 |
| 360448 | 3474 | 1306 | 2.66x | 7424->1894 | 9856->6198 | 1424->198 |
| 376832 | 3663 | 1365 | 2.68x | 7760->1978 | 10304->6480 | 1488->206 |
| 393216 | 3811 | 1396 | 2.73x | 8096->2063 | 10752->6762 | 1552->214 |
| 409600 | 3975 | 1478 | 2.69x | 8432->2148 | 11200->7044 | 1616->222 |
| 425984 | 4116 | 1534 | 2.68x | 8768->2232 | 11648->7326 | 1680->230 |
| 442368 | 4259 | 1590 | 2.68x | 9104->2317 | 12096->7607 | 1744->239 |
| 458752 | 4479 | 1617 | 2.77x | 9440->2402 | 12544->7889 | 1808->247 |

## Conclusion

**The TQ4 speedup rises steeply, then converges.** It climbs from 1.85x at
`n=16384` to ~2.7x by `n~=96k`, then plateaus in a 2.64x-2.83x band for the rest
of the sweep (no sustained upward trend out to `n=458752`). So as the `O(nl)`
communication payload grows, TQ4's benefit saturates rather than growing without
bound: once the compressed reduce dominates total time, both NONE and TQ4 scale
linearly in `n`, fixing the ratio at the inverse of the reduce's compression
factor plus the remaining uncompressed/compute terms.

This is consistent with the payload numbers: InfiniBand payload (the inter-node
reduce) is cut ~3.3x at small `n` and ~7x at large `n` (the per-step `O(nl)` part
shrinks faster than fixed overheads), while NVLink payload is only cut ~1.6x
(intra-node traffic includes uncompressed pieces). The end-to-end ~2.7x reflects
the blend of these compressed and uncompressed stages.

Note this is a much larger speedup than the ~1.6x seen in the balanced
matrix-size scaling (where `m` and `n` grow together): holding `m` fixed makes
the `O(nl)` reduce payload the dominant cost sooner, so communication
compression has more to bite on.

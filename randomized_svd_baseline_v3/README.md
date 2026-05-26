# Randomized SVD Baseline v3

This directory contains the MPI multi-node version of
`randomized_svd_baseline_v2`.

## Current Scope

v3 uses one MPI rank per node. Each rank controls the GPUs visible on that
node through the same CUDA/cuBLAS/cuSOLVER path used in v2.

For a 2-node, 8-GPU-per-node run:

```text
MPI rank 0 -> node 0, local GPUs 0..7
MPI rank 1 -> node 1, local GPUs 0..7
global GPUs = 16
```

The global matrix is row-split across all MPI ranks and all local GPUs:

```text
A = [A_0; A_1; ...; A_15]
```

## Communication Model

The v3 version uses host MPI for cross-node collectives:

1. Each local GPU computes `R_i`.
2. Each MPI rank gathers its local `R_i` blocks to host.
3. MPI rank 0 gathers all `R_i`, runs the TSQR reduction, and broadcasts the
   stacked `T_i` factors.
4. Each local GPU computes `B_i = Q_i^T A_i`.
5. Each rank performs node-local optional `B_i` compression/reconstruction and
   node-local `B` accumulation.
6. For `--compress-b-mode tq`, `tq-qjl`, or `lowbit`, each rank compresses its
   node-local `B` again, MPI gathers only the compressed payload and metadata,
   and rank 0 decodes/adds the payloads on GPU 0.
7. For `--compress-b-mode none`, v3 falls back to the FP32 MPI reduction.
8. Rank 0 computes `SVD(B)`.

This is the compressed MPI collective version of the baseline. It is still a
root-gather/decode implementation rather than a tuned tree all-reduce, but the
large cross-node `B` payload is compressed in the TQ/TQ+QJL/lowbit modes.

The output reports two payload groups:

```text
reduce_B_*  node-local sum of per-GPU B_i payloads
mpi_B_*     cross-node rank-local B payload sent through MPI
```

For compressed modes, `mpi_B_collective_mode` should be
`compressed_gather_decode`.

## Build

On Taiwania 2, use an MPI environment before building:

```bash
module load cuda/12.8
module load ucx/1.14.1
module load openmpi/5.0.2_ucx1.14.1_cuda12.3
make
```

## Run

The main Slurm script is:

```bash
sbatch run_randomized_svd_multigpu_v3_tq_bit_curve_16gpu.slurm
```

It requests:

```text
2 nodes
1 MPI rank per node
8 GPUs per node
16 GPUs total
```

## Current 16-GPU Result

Base run:

```text
job_id=927251
m=32768 n=8192 k=256 l=320
nodes=2 ranks=2 gpus_per_rank=8 global_gpus=16
repeat=5, warm average excludes the first cold-start repeat
```

| mode | warm pipeline avg | node-local B payload | MPI B payload | MPI B collective | B relative error |
| --- | ---: | ---: | ---: | --- | ---: |
| none | 64.5616 ms | 160 MiB | 20 MiB | fp32_reduce | skipped |
| tq 4-bit | 50.0871 ms | 32.0001 MiB | 4.00002 MiB | compressed_gather_decode | 0.175305 |
| tq 2-bit | 47.1681 ms | 16.0001 MiB | 2.00002 MiB | compressed_gather_decode | 0.952533 |

Compared with `none`, this run gives about 22.4% lower warm pipeline time for
TQ4 and about 26.9% lower warm pipeline time for TQ2.

Larger matrix run:

```text
job_id=927252
m=65536 n=16384 k=256 l=320
nodes=2 ranks=2 gpus_per_rank=8 global_gpus=16
repeat=20, warm average excludes the first cold-start repeat
```

| mode | warm pipeline avg | node-local B payload | MPI B payload | MPI B collective | B relative error |
| --- | ---: | ---: | ---: | --- | ---: |
| none | 90.4434 ms | 320 MiB | 40 MiB | fp32_reduce | skipped |
| tq 4-bit | 73.2406 ms | 64.0001 MiB | 8.00002 MiB | compressed_gather_decode | 0.180200 |
| tq 2-bit | 66.5009 ms | 32.0001 MiB | 4.00002 MiB | compressed_gather_decode | 0.960169 |

Compared with `none`, the larger run gives about 19.0% lower warm pipeline time
for TQ4 and about 26.5% lower warm pipeline time for TQ2.

After removing the final synchronization from the shared TQ-only column
quantization path, the same large run was repeated:

```text
job_id=927253
m=65536 n=16384 k=256 l=320
nodes=2 ranks=2 gpus_per_rank=8 global_gpus=16
repeat=20
```

| mode | warm pipeline avg before | warm pipeline avg after | change |
| --- | ---: | ---: | ---: |
| none | 90.4434 ms | 90.4659 ms | +0.02 ms |
| tq 4-bit | 73.2406 ms | 72.1659 ms | -1.07 ms |
| tq 2-bit | 66.5009 ms | 66.2885 ms | -0.21 ms |

This optimization affects the shared `turboquant/` TQ path, so it applies to
both v2 and v3. The small but isolated speedup suggests the next optimization
step should use Nsight Compute on the TQ FWHT/pack/decode kernels rather than
only adjusting host-side scheduling.

The next TQ-only optimization fused the column TQ decode-add path. Previously,
decode-add first expanded packed integer codes into a global `d_work` buffer and
then launched a second kernel for inverse FWHT plus accumulation. The fused path
loads packed codes directly into shared memory, dequantizes there, runs inverse
FWHT, and accumulates into `B`.

```text
job_id=927254
m=65536 n=16384 k=256 l=320
nodes=2 ranks=2 gpus_per_rank=8 global_gpus=16
repeat=20
```

| mode | after sync removal | after fused decode-add | change |
| --- | ---: | ---: | ---: |
| none | 90.4659 ms | 90.2220 ms | -0.24 ms |
| tq 4-bit | 72.1659 ms | 71.4897 ms | -0.68 ms |
| tq 2-bit | 66.2885 ms | 64.8231 ms | -1.47 ms |

The measured B relative errors stayed the same:

| mode | B relative error |
| --- | ---: |
| tq 4-bit | 0.180200 |
| tq 2-bit | 0.960169 |

## GPU Scale Limit

A 4-node / 32-GPU probe was submitted as job `927250`. Slurm kept it pending
with:

```text
QOSMaxNodePerJobLimit
```

So the current practical maximum for one job is 2 nodes, i.e. 16 V100 GPUs with
the current `nycugpu_queue` / `contest_v100` policy.

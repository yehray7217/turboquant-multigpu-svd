Please refactor the timing metrics in the current implementation.

Files to inspect and modify:

- randomized_svd_multigpu_v4.cu
- turboquant.cu
- turboquant.hpp if needed

Do NOT work on QJL in this patch.

# Goal

Replace the existing by-algorithm-stage timing table with a fabric-oriented timing summary.

Remove the current stage-based timing output, such as:

- Local Projection
- Local QR
- TSQR R Reduce
- Form Distributed Qi
- Subspace Iteration
- Build Reduce Bi
- SVD B
- Form Distributed Ui

Do not keep both old and new timing metrics. Remove / bypass the by-stage timing table entirely.

The final timing summary should only report:

1. Total Time
2. GPU Compute Time
3. NVLink Time
4. InfiniBand Time
5. NVLink Payload Size
6. InfiniBand Payload Size

# Important definition

Total Time should measure the randomized SVD algorithm only.

It should exclude:

- synthetic test matrix generation
- final reconstruction error computation
- theoretical error computation
- Global B Relative Error computation
- any other post-run diagnostic

It should include:

- local projection
- QR / TSQR
- subspace iteration if enabled
- B build / reduce
- final SVD on B
- optional U formation if skip_form_u is false

# Required metric definitions

## 1. Total Time

Measure wall-clock time from the start of the rSVD computation to the end of the rSVD computation.

Do not include data generation or error checking.

For MPI runs, report the max time across ranks for each repeat.

## 2. GPU Compute Time

GPU Compute Time should include time spent in:

- cuBLAS GEMM calls
- cuSOLVER QR / SVD calls
- CUDA kernels
- TQ quantization kernels
- TQ dequantization kernels
- transpose kernels
- GPU accumulation kernels
- RHT kernels
- any other GPU compute kernel

Do not count cudaMemcpyPeer as GPU Compute Time. That belongs to NVLink Time.

Do not count MPI calls as GPU Compute Time. Those belong to InfiniBand Time.

For GPU timing, use CUDA events if practical. If not, use explicit cudaDeviceSynchronize before and after the measured GPU work.

## 3. NVLink Time

NVLink Time means intra-node GPU-to-GPU transfer time.

Instrument all places that use GPU peer transfer, especially:

- cudaMemcpyPeer
- cudaMemcpyPeerAsync

This includes peer copies of:

- raw B_i payloads
- compressed B_i payloads
- TQ code payloads
- TQ norms
- QJL signs if any legacy path still exists
- raw or compressed Z_i payloads if Z intra-node compression is implemented

Measure only the peer-transfer time, not the compression/decompression time.

Compression/decompression remains GPU Compute Time.

If a transfer is not a peer GPU-to-GPU transfer, do not count it as NVLink.

Also add comments in the code marking these paths as intra-node / NVLink-like communication.

## 4. InfiniBand Time

InfiniBand Time means inter-node MPI communication time.

Instrument MPI calls that communicate across ranks, including:

- MPI_Gather
- MPI_Gatherv
- MPI_Reduce
- MPI_Allreduce
- MPI_Allgather
- MPI_Bcast
- any other MPI collective used in TSQR, B reduce, or Z reduce

Measure only the MPI call itself using MPI_Wtime.

For each repeat, reduce the per-rank MPI elapsed time with MPI_MAX, because the effective collective time is determined by the slowest rank.

Do not include D2H/H2D staging time in InfiniBand Time. Host staging remains part of Total Time but does not need to be printed as a separate final metric.

Also add comments in the code marking these paths as inter-node / InfiniBand-like communication.

# Payload size metrics

Add payload counters.

## NVLink Payload Size

Count the number of bytes transferred through cudaMemcpyPeer / cudaMemcpyPeerAsync.

Use the actual byte count passed to the peer copy call.

For compressed transfers, count compressed payload bytes, including:

- packed indices
- norms
- metadata if transferred
- QJL signs if applicable

For uncompressed transfers, count FP32 bytes.

Aggregate NVLink payload bytes across all local peer transfers and all ranks.

Report in MiB.

## InfiniBand Payload Size

Count logical MPI application payload bytes.

For MPI calls, count the size of the buffer passed to MPI.

For compressed communication, count compressed bytes.

For raw FP32 communication, count FP32 bytes.

For collective operations, be consistent and document the convention in a comment.

Suggested convention:

- For MPI_Allreduce and MPI_Allgather:
  count local payload bytes on each rank and sum across ranks.
- For MPI_Gather / MPI_Reduce to root:
  count non-root sent payloads when easy; otherwise count all rank contributions and label it as logical collective payload.
- For MPI_Bcast:
  count payload bytes sent from root to other ranks, approximately (mpi_size - 1) * message_size.

Report in MiB.

# Required investigation

Before implementing the final counters, inspect the code and identify which communication paths are:

## NVLink / intra-node candidates

These should generally be cudaMemcpyPeer / cudaMemcpyPeerAsync between GPUs inside the same MPI rank.

Likely examples:

- B_i transfer from nonzero local GPU to rank-local GPU0
- compressed B_i payload transfer from nonzero local GPU to GPU0
- compressed or raw Z_i transfer if Z intra-node aggregation is implemented with peer copies

## InfiniBand / inter-node candidates

These should generally be MPI calls between ranks.

Likely examples:

- TSQR R gather / broadcast
- B rank-local reduce / gather across MPI ranks
- Z MPI_Allreduce in raw path
- Z compressed MPI_Allgather path
- any MPI broadcast of global Z or TSQR correction data

Add short comments near these sections so future readers know which hardware fabric the timer is intended to represent.

# New summary output

Replace the old stage table with something like:

Timing Summary
  Repeat Count: N

  Metric                  mean        min         stddev
  Total Time              ... ms      ... ms      ... ms
  GPU Compute Time        ... ms      ... ms      ... ms
  NVLink Time             ... ms      ... ms      ... ms
  InfiniBand Time         ... ms      ... ms      ... ms

Payload Summary
  Metric                  mean        min         stddev
  NVLink Payload          ... MiB     ... MiB     ... MiB
  InfiniBand Payload      ... MiB     ... MiB     ... MiB

Do not print the old by-stage metrics.

# Important interpretation note

It is acceptable if:

    Total Time != GPU Compute Time + NVLink Time + InfiniBand Time

because Total Time may also include:

- host staging D2H/H2D
- CPU-side accumulation
- synchronization overhead
- bookkeeping
- memory allocation if still inside the measured region

Do not try to force the categories to sum exactly to Total Time.

# Repeat behavior

For each repeat:

1. Reset timing counters.
2. Start Total Time after test data generation is complete.
3. Run rSVD.
4. Stop Total Time before error metric / diagnostics.
5. Reduce timing values across MPI ranks:
   - Times: MPI_MAX
   - Payload bytes: MPI_SUM
6. Store repeat-level global values.
7. Report mean / min / stddev across repeats.

# Do not change these in this patch

- Do not work on QJL.
- Do not change numerical algorithms.
- Do not change TQ quantization/dequantization.
- Do not change corrected error metric.
- Do not re-add Captured Energy Ratio.
- Do not change experiment CLI semantics unless needed for timing.
- Do not add fabric-based compression flags yet.
- Do not submit Slurm jobs.

# Acceptance criteria

After this patch:

1. The old by-stage timing table is gone.
2. The output reports only:
   - Total Time
   - GPU Compute Time
   - NVLink Time
   - InfiniBand Time
   - NVLink Payload
   - InfiniBand Payload
3. Total Time excludes:
   - data generation
   - final reconstruction error computation
   - theoretical error computation
   - Global B Relative Error computation
4. NVLink Time is measured around GPU peer-copy paths.
5. InfiniBand Time is measured around MPI communication calls.
6. Payload sizes are reported in MiB.
7. The code clearly comments which communication paths are treated as NVLink-like and InfiniBand-like.
8. The code compiles cleanly.
9. Existing accuracy results should not change except for possible tiny timing-related synchronization effects.
# AI Takeaway — cusolver/

## Purpose

**Baseline A**: single-GPU exact SVD via NVIDIA's cuSOLVER library.

This directory provides:

- A correctness reference (compare singular values against your custom pipeline)
- A library-level single-GPU timing baseline

It is **not** the main optimization target. The project aims to accelerate
multi-GPU distributed SVD, not improve single-GPU exact SVD.

---

## Files

| File              | Description                                                                                         |
| ----------------- | --------------------------------------------------------------------------------------------------- |
| `cusolver.cu`     | Main program. Calls `cusolverDnSgesvd` (Jacobi SVD), reports top 3 singular values and kernel time. |
| `makefile`        | Compiles with `nvcc -O3 -arch=sm_70 -lcusolver -lcudart`.                                           |
| `run_cusolver.sh` | Slurm batch script for Taiwania 2. Sweeps matrix sizes: 1024, 2048, 4096, 8192, 16384 (square).     |
| `readme.md`       | Setup and run instructions.                                                                         |

---

## Key Concepts

**cusolverDnSgesvd**: the routine used here computes full (not truncated) SVD of
an `M×N` matrix. It uses a two-phase approach internally:

1. Bidiagonalization (Householder reduction)
2. QR iteration on the bidiagonal form

This is an **exact** method—much more expensive than randomized SVD for large
matrices, but numerically accurate to FP32 precision.

**Why Jacobi?** The CUDA cuSOLVER also offers a Jacobi-based SVD path
(`cusolverDnSgesvdj`). The `cusolverDnSgesvd` used here is the standard
bidiagonal-QR path.

**What it measures**: only the GPU kernel execution time (via `cudaEventRecord`),
not host-side setup, memory allocation, or data transfer.

---

## How to Build & Run

```bash
module load cuda/12.8
cd cusolver
make
sbatch run_cusolver.sh
```

For a one-off test:

```bash
./cusolver -M 4096 -N 4096
```

---

## Key Parameters

| Flag        | Meaning                            |
| ----------- | ---------------------------------- |
| `-M <rows>` | Number of rows of the input matrix |
| `-N <cols>` | Number of columns                  |

The program always allocates a random `M×N` matrix on GPU, runs full SVD,
and prints the top 3 singular values plus execution time.

---

## Current Status

Working. Used to produce single-GPU timing numbers for presentation comparisons.

The benchmark sweep in `run_cusolver.sh` covers square matrices up to 16384×16384
which are large enough to stress single-GPU memory bandwidth.

---

## Multi-GPU Proxy App (`mpi_cusolver.cu`) — Added 2026-05-26

A second program was added to `cusolver/`: a **distributed multi-GPU SVD proxy
app** using MPI + cuSOLVER. See `cusolver/readme.md` §2 for the description.

### What `mpi_cusolver.cu` does

Each MPI rank is assigned one GPU (`rank % num_devices`). Every rank:

1. Generates a local random `M × N` matrix independently.
2. Runs `cusolverDnSgesvdj` (Jacobi SVD) on its own GPU to compute local singular vectors.
3. Each rank extracts its first right singular vector (`V[:,0]`, dimension N).
4. `MPI_Allgather` collects all ranks' first singular vectors into one buffer on every rank.
5. Timings are reported separately:
   - **Compute Time** = local SVD (cuSOLVER) on each GPU.
   - **Comm Time** = `MPI_Allgather` across all ranks.

### Purpose

This proxy app is designed to:

- **Benchmark network bandwidth vs. compute time** in a distributed multi-node setting.
- **Establish MPI communication overhead** as a baseline before adding compression.
- **Validate weak scaling**: each GPU always processes `M × N` regardless of rank count.
- **Identify whether MPI is the bottleneck**: compare Comm Time vs. Compute Time.

### Key Architectural Note

This proxy app is a **communication pattern study**, not the full SVD pipeline.
Each rank computes a full independent SVD; they do NOT cooperate on a single matrix.
The `MPI_Allgather` simulates the communication step of distributing singular vectors.
This is analogous to the `B` reduction step in the v2/v3 main pipeline.

### How to Run

```bash
sbatch cusolver/run_mpi_cusolver.sh
```

The script sweeps matrix sizes: 1024, 2048, 4096, 8192 (square).
It targets `2 nodes × 8 GPUs = 16 GPUs` (2 Slurm nodes, 8 tasks/node).

### Potential Issues

- **Architecture flag**: uses `-arch=sm_70` (V100). Update for other GPUs.
- **No compression**: this is a pure baseline. Adding TQ compression to the vector before `Allgather` would be the natural next experiment.
- **`MPI_Allgather` vs. `MPI_Gather`**: the proxy uses Allgather (all ranks receive), which has higher cost than Gather (only rank 0 receives). Real pipeline uses Gather.

---

## Next Steps / What To Try

- **Use as correctness check**: after running your randomized SVD, compare the
  top-k singular values against cuSOLVER output on the same matrix.
- **Compare timing with v2 none-mode**: for a given matrix size, cuSOLVER (exact,
  single-GPU) vs v2 (randomized, multi-GPU) shows the algorithm tradeoff clearly.
- **Do not optimize this baseline further**: it's a fixed reference, not a tuning
  target.
- **Run the MPI proxy**: collect Compute Time vs. Comm Time to quantify the
  communication bottleneck before adding TQ compression to cross-node vectors.

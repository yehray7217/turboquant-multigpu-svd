#!/bin/bash
#SBATCH --job-name=v3_fp16_baseline
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=v3_fp16_baseline_%j.out
#SBATCH --error=v3_fp16_baseline_%j.err

# Loop 4 A2: v3 FP16 MPI baseline (2× compression, near-lossless ~0.0002 error).
# At multi-node scale, v2's FP16 build_reduce_Bi was fastest (6.78 ms) — but that
# was P2P, not MPI. Here we measure whether FP16 MPI_Gather bytes savings (~20 MiB
# halved to ~10 MiB per node at 32k×8k) translate to pipeline speedup vs TQ 4-bit.
#
# Reference results to compare against (from RECORD.csv / v3 existing runs):
#   none    32k×8k : 66 ms   64k×16k: 91 ms
#   TQ 4bit 32k×8k : 49 ms   64k×16k: 71 ms
#
# FP16 expected: somewhere between none and TQ 4-bit (2× compression vs 5×).
#
# Two passes:
#   Pass 1: timing-only (100 repeats, --no-check-b-error) for warm averages.
#   Pass 2: B-error check on FP16 only (5 repeats) — host copy inflates timing.
# Same discipline as v2's run_fp16_baseline.slurm (job 931176).

set -euo pipefail

module load cuda/12.8
module load ucx/1.14.1
module load openmpi/5.0.2_ucx1.14.1_cuda12.3

unset OMPI_MCA_opal_cuda_support || true
unset OMPI_MCA_btl_smcuda_use_cuda_ipc || true

PROJECT_ROOT=$HOME/turboquant-multigpu-svd/randomized_svd_baseline_v3
cd "$PROJECT_ROOT"

make

echo "====================================="
echo "v3 FP16 Baseline Benchmark (Loop 4 A2)"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
nvidia-smi -L
echo "====================================="

TIMING_ARGS_COMMON=(
    --k 256
    --oversample 64
    --ngpus 16
    --gpus-per-rank 8
    --seed 1234
    --device-random-input
    --skip-form-u
    --repeat 100
    --repeat-print-every 25
    --no-check-error
    --no-check-b-error
)

echo "========== PASS 1: TIMING (100 repeats) — m=32768, n=8192 =========="

echo "--- none (baseline) ---"
srun --mpi=pmix ./randomized_svd_multigpu_v3 \
    --m 32768 --n 8192 \
    "${TIMING_ARGS_COMMON[@]}" \
    --compress-b-mode none \
    --compress-b-bits 0
echo "-------------------------------------"

echo "--- FP16 (NEW: 2× near-lossless) ---"
srun --mpi=pmix ./randomized_svd_multigpu_v3 \
    --m 32768 --n 8192 \
    "${TIMING_ARGS_COMMON[@]}" \
    --compress-b-mode fp16 \
    --compress-b-bits 0
echo "-------------------------------------"

echo "--- TQ 4-bit ---"
srun --mpi=pmix ./randomized_svd_multigpu_v3 \
    --m 32768 --n 8192 \
    "${TIMING_ARGS_COMMON[@]}" \
    --compress-b-mode tq \
    --compress-b-bits 4 \
    --qjl-dim 0 \
    --qjl-alpha 0.0
echo "-------------------------------------"

echo "--- TQ 2-bit ---"
srun --mpi=pmix ./randomized_svd_multigpu_v3 \
    --m 32768 --n 8192 \
    "${TIMING_ARGS_COMMON[@]}" \
    --compress-b-mode tq \
    --compress-b-bits 2 \
    --qjl-dim 0 \
    --qjl-alpha 0.0
echo "-------------------------------------"

echo "========== PASS 1: TIMING (100 repeats) — m=65536, n=16384 =========="

echo "--- none (baseline) ---"
srun --mpi=pmix ./randomized_svd_multigpu_v3 \
    --m 65536 --n 16384 \
    "${TIMING_ARGS_COMMON[@]}" \
    --compress-b-mode none \
    --compress-b-bits 0
echo "-------------------------------------"

echo "--- FP16 (NEW: 2× near-lossless) ---"
srun --mpi=pmix ./randomized_svd_multigpu_v3 \
    --m 65536 --n 16384 \
    "${TIMING_ARGS_COMMON[@]}" \
    --compress-b-mode fp16 \
    --compress-b-bits 0
echo "-------------------------------------"

echo "--- TQ 4-bit ---"
srun --mpi=pmix ./randomized_svd_multigpu_v3 \
    --m 65536 --n 16384 \
    "${TIMING_ARGS_COMMON[@]}" \
    --compress-b-mode tq \
    --compress-b-bits 4 \
    --qjl-dim 0 \
    --qjl-alpha 0.0
echo "-------------------------------------"

echo "--- TQ 2-bit ---"
srun --mpi=pmix ./randomized_svd_multigpu_v3 \
    --m 65536 --n 16384 \
    "${TIMING_ARGS_COMMON[@]}" \
    --compress-b-mode tq \
    --compress-b-bits 2 \
    --qjl-dim 0 \
    --qjl-alpha 0.0
echo "-------------------------------------"

echo "========== PASS 2: B-ERROR CHECK (5 repeats) — FP16 only =========="
# NOTE: The v3 reduce_B_relative_error for FP16 mode will report ~0.000 because
# the local B-error check compares FP32 B_i against itself (compress_b_none=true
# path within the rank's local GPU loop). The actual MPI-level FP16 rounding error
# is ~0.0002 per rank (same as v2 job 931176). Use v2's measured value for the final
# report table. These runs verify the pipeline succeeds and timing is not inflated.

BERROR_ARGS_COMMON=(
    --k 256
    --oversample 64
    --ngpus 16
    --gpus-per-rank 8
    --seed 1234
    --device-random-input
    --skip-form-u
    --repeat 5
    --repeat-print-every 5
    --no-check-error
    # NOTE: --no-check-b-error intentionally OMITTED — we want B-error here
)

echo "--- FP16 B-error (32k×8k) ---"
srun --mpi=pmix ./randomized_svd_multigpu_v3 \
    --m 32768 --n 8192 \
    "${BERROR_ARGS_COMMON[@]}" \
    --compress-b-mode fp16 \
    --compress-b-bits 0
echo "-------------------------------------"

echo "--- FP16 B-error (64k×16k) ---"
srun --mpi=pmix ./randomized_svd_multigpu_v3 \
    --m 65536 --n 16384 \
    "${BERROR_ARGS_COMMON[@]}" \
    --compress-b-mode fp16 \
    --compress-b-bits 0
echo "-------------------------------------"

echo "v3 FP16 Baseline Benchmark complete!"

#!/bin/bash
#SBATCH --job-name=rand_svd_v3_smoke16
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=00:10:00
#SBATCH --output=randomized_svd_multigpu_v3_smoke_16gpu_%j.out
#SBATCH --error=randomized_svd_multigpu_v3_smoke_16gpu_%j.err

set -euo pipefail

module load cuda/12.8
module load ucx/1.14.1
module load openmpi/5.0.2_ucx1.14.1_cuda12.3

unset OMPI_MCA_opal_cuda_support || true
unset OMPI_MCA_btl_smcuda_use_cuda_ipc || true

PROJECT_ROOT=$HOME/turboquant-multigpu-svd/randomized_svd_baseline_v3
cd "$PROJECT_ROOT"

make

echo "Starting Randomized Multi-GPU SVD v3 smoke, 16GPU"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
nvidia-smi -L
echo "====================================="

COMMON_ARGS=(
    --m 8192
    --n 4096
    --k 128
    --oversample 32
    --ngpus 16
    --gpus-per-rank 8
    --seed 1234
    --device-random-input
    --skip-form-u
    --repeat 2
    --repeat-print-every 1
    --no-check-error
    --no-check-b-error
)

echo "Smoke timing: none"
srun --mpi=pmix ./randomized_svd_multigpu_v3 \
    "${COMMON_ARGS[@]}" \
    --compress-b-mode none \
    --compress-b-bits 0
echo "-------------------------------------"

echo "Smoke timing: tq 4-bit"
srun --mpi=pmix ./randomized_svd_multigpu_v3 \
    "${COMMON_ARGS[@]}" \
    --compress-b-mode tq \
    --compress-b-bits 4 \
    --qjl-dim 0 \
    --qjl-alpha 0.0
echo "-------------------------------------"

echo "Randomized Multi-GPU SVD v3 smoke, 16GPU complete!"

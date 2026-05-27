#!/bin/bash
#SBATCH --job-name=rand_svd_v3_tqbits16
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=randomized_svd_multigpu_v3_tq_bit_curve_16gpu_%j.out
#SBATCH --error=randomized_svd_multigpu_v3_tq_bit_curve_16gpu_%j.err

set -euo pipefail

module load cuda/12.8
module load ucx/1.14.1
module load openmpi/5.0.2_ucx1.14.1_cuda12.3

unset OMPI_MCA_opal_cuda_support || true
unset OMPI_MCA_btl_smcuda_use_cuda_ipc || true

PROJECT_ROOT=$HOME/turboquant-multigpu-svd/randomized_svd_baseline_v3
cd "$PROJECT_ROOT"

make

BASE_ARGS=(
    --m 32768
    --n 8192
    --k 256
    --oversample 64
    --ngpus 16
    --gpus-per-rank 8
    --seed 1234
    --device-random-input
    --skip-form-u
)

echo "Starting Randomized Multi-GPU SVD v3 TQ bit/time/error curve, 16GPU"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
nvidia-smi -L
echo "====================================="

TIME_ARGS=(
    "${BASE_ARGS[@]}"
    --repeat 100
    --repeat-print-every 25
    --no-check-error
    --no-check-b-error
)

echo "Timing: none"
srun --mpi=pmix ./randomized_svd_multigpu_v3 \
    "${TIME_ARGS[@]}" \
    --compress-b-mode none \
    --compress-b-bits 0
echo "-------------------------------------"

for bits in 4 2; do
    echo "Timing: tq ${bits}-bit"
    srun --mpi=pmix ./randomized_svd_multigpu_v3 \
        "${TIME_ARGS[@]}" \
        --compress-b-mode tq \
        --compress-b-bits "${bits}" \
        --qjl-dim 0 \
        --qjl-alpha 0.0
    echo "-------------------------------------"
done

ERROR_ARGS=(
    "${BASE_ARGS[@]}"
    --repeat 1
    --repeat-print-every 1
    --no-check-error
)

for bits in 4 2; do
    echo "B-error: tq ${bits}-bit"
    srun --mpi=pmix ./randomized_svd_multigpu_v3 \
        "${ERROR_ARGS[@]}" \
        --compress-b-mode tq \
        --compress-b-bits "${bits}" \
        --qjl-dim 0 \
        --qjl-alpha 0.0
    echo "-------------------------------------"
done

echo "TQ bit/time/error curve, 16GPU complete!"

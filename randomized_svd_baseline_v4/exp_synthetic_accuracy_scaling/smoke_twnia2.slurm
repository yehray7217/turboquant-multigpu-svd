#!/bin/bash
#SBATCH --job-name=v4_smoke
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=00:20:00
#SBATCH --output=$HOME/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_synthetic_accuracy_scaling/output_logs/smoke_%j.out
#SBATCH --error=$HOME/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_synthetic_accuracy_scaling/error_logs/smoke_%j.err

set -euo pipefail

module purge
module load cuda/12.8
module load openmpi/5.0.2_ucx1.14.1_cuda12.3

unset OMPI_MCA_opal_cuda_support || true
unset OMPI_MCA_btl_smcuda_use_cuda_ipc || true

PROJECT_ROOT=$HOME/turboquant-multigpu-svd/randomized_svd_baseline_v4
BIN="${PROJECT_ROOT}/.build/randomized_svd_multigpu_v4"

cd "$PROJECT_ROOT"
make

echo "Smoke test"
echo "Date: $(date)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
nvidia-smi -L
echo "====================================="

base_args=(
    --m 8192 --n 2048 --k 250 --oversample 6
    --ngpus 16 --gpus-per-rank 8 --seed 1234
    --spectrum-decay-mode polynomial --spectrum-decay-param 0.6
    --spectrum-rank 2048 --subspace-iter 1
    --device-random-input --skip-form-u --summary-only
)

echo "--- none (timing) ---"
srun --mpi=pmix "$BIN" "${base_args[@]}" --compress-b-mode none --compress-b-bits 0 --compress-subspace-mode none --compress-subspace-bits 0 --repeat 3 --no-check-error

echo "--- none (final-error) ---"
srun --mpi=pmix "$BIN" "${base_args[@]}" --compress-b-mode none --compress-b-bits 0 --compress-subspace-mode none --compress-subspace-bits 0 --repeat 3

echo "--- tq8 (final-error + b-error) ---"
srun --mpi=pmix "$BIN" "${base_args[@]}" --compress-b-mode tq --compress-b-bits 8 --compress-subspace-mode tq --compress-subspace-bits 8 --repeat 3 --check-b-error

echo "Smoke complete."

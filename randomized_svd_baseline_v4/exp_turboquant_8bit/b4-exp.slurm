#!/bin/bash
#SBATCH --job-name=rsvd_v4_tq4_exp
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=/work/pbr03617/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_turboquant_8bit/output_logs/b4-exp.out
#SBATCH --error=/work/pbr03617/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_turboquant_8bit/error_logs/b4-exp.err

set -euo pipefail

module load cuda/12.8
module load ucx/1.14.1
module load openmpi/5.0.2_ucx1.14.1_cuda12.3

unset OMPI_MCA_opal_cuda_support || true
unset OMPI_MCA_btl_smcuda_use_cuda_ipc || true

PROJECT_ROOT=/work/pbr03617/turboquant-multigpu-svd/randomized_svd_baseline_v4
BIN="${PROJECT_ROOT}/.build/randomized_svd_multigpu_v4"
cd "$PROJECT_ROOT"

make
cd "${PROJECT_ROOT}/exp_turboquant_8bit"

BASE_ARGS=(
    --m 65536
    --n 16384
    --k 250
    --oversample 6
    --ngpus 16
    --gpus-per-rank 8
    --seed 1234
    --spectrum-decay-mode polynomial
    --spectrum-decay-param 0.6
    --spectrum-rank 8192
    --subspace-iter 1
    --device-random-input
    --skip-form-u
    --compress-b-mode tq
    --compress-b-bits 4
    --compress-subspace-mode tq
    --compress-subspace-bits 4
    --summary-only
)

echo "Starting experiment group: TQ 4-bit"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
nvidia-smi -L
echo "====================================="

echo "Timing: experiment"
srun --mpi=pmix "$BIN" \
    "${BASE_ARGS[@]}" \
    --repeat 50 \
    --no-check-error
echo "-------------------------------------"

echo "Final-error: experiment"
srun --mpi=pmix "$BIN" \
    "${BASE_ARGS[@]}" \
    --repeat 50

echo "Experiment group complete!"

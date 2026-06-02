#!/bin/bash
#SBATCH --job-name=v4_tq_pack4_eval
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=output_logs/v4_tq_pack4_eval_%j.out
#SBATCH --error=error_logs/v4_tq_pack4_eval_%j.err

set -euo pipefail

module load cuda/12.8
module load ucx/1.14.1
module load openmpi/5.0.2_ucx1.14.1_cuda12.3

unset OMPI_MCA_opal_cuda_support || true
unset OMPI_MCA_btl_smcuda_use_cuda_ipc || true

PROJECT_ROOT="$HOME/turboquant-multigpu-svd/randomized_svd_baseline_v4"
BIN="${PROJECT_ROOT}/.build/randomized_svd_multigpu_v4"

cd "$PROJECT_ROOT"
make

cd "${PROJECT_ROOT}/exp_cuda_opt_round1"

COMMON_ARGS=(
    --m 65536
    --n 16384
    --k 250
    --oversample 6
    --ngpus 8
    --gpus-per-rank 8
    --seed 1234
    --spectrum-decay-mode polynomial
    --spectrum-decay-param 0.6
    --spectrum-rank 8192
    --subspace-iter 1
    --device-random-input
    --skip-form-u
    --summary-only
)

run_case() {
    local label="$1"
    local b_mode="$2"
    local b_bits="$3"
    local z_mode="$4"
    local z_bits="$5"

    echo "====================================="
    echo "CASE=${label}"
    echo "Date: $(date)"

    echo "Timing: ${label}"
    srun --mpi=pmix "$BIN" \
        "${COMMON_ARGS[@]}" \
        --compress-b-mode "${b_mode}" \
        --compress-b-bits "${b_bits}" \
        --compress-subspace-mode "${z_mode}" \
        --compress-subspace-bits "${z_bits}" \
        --repeat 30 \
        --no-check-error

    echo "Final-error: ${label}"
    srun --mpi=pmix "$BIN" \
        "${COMMON_ARGS[@]}" \
        --compress-b-mode "${b_mode}" \
        --compress-b-bits "${b_bits}" \
        --compress-subspace-mode "${z_mode}" \
        --compress-subspace-bits "${z_bits}" \
        --repeat 30
}

echo "v4 CUDA optimization round 1: TQ4/TQ2 specialized bitpack evaluation"
echo "Host: $(hostname)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
nvidia-smi -L

run_case "none" "none" 0 "none" 0
run_case "TQ4" "tq" 4 "tq" 4

echo "Round 1 evaluation complete."

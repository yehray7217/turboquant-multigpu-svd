#!/bin/bash
#SBATCH --job-name=v4_syn_acc
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=$HOME/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_synthetic_accuracy_scaling/output_logs/accuracy_sweep_%j.out
#SBATCH --error=$HOME/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_synthetic_accuracy_scaling/error_logs/accuracy_sweep_%j.err

set -euo pipefail

module purge
module load cuda/12.8
module load openmpi/5.0.2_ucx1.14.1_cuda12.3

unset OMPI_MCA_opal_cuda_support || true
unset OMPI_MCA_btl_smcuda_use_cuda_ipc || true

PROJECT_ROOT=$HOME/turboquant-multigpu-svd/randomized_svd_baseline_v4
EXP_DIR="${PROJECT_ROOT}/exp_synthetic_accuracy_scaling"
BIN="${PROJECT_ROOT}/.build/randomized_svd_multigpu_v4"

REPEAT="${REPEAT:-50}"
CHECK_B_ERROR="${CHECK_B_ERROR:-1}"
PVALS="${PVALS:-0.4 0.6 0.8 1.0}"

cd "$PROJECT_ROOT"
make
cd "$EXP_DIR"

echo "Synthetic accuracy sweep"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "REPEAT=${REPEAT} CHECK_B_ERROR=${CHECK_B_ERROR}"
nvidia-smi -L
echo "====================================="

run_case() {
    local p="$1"
    local label="$2"
    local b_mode="$3"
    local b_bits="$4"
    local z_mode="$5"
    local z_bits="$6"

    local -a base_args=(
        --m 32768
        --n 8192
        --k 250
        --oversample 6
        --ngpus 16
        --gpus-per-rank 8
        --seed 1234
        --spectrum-decay-mode polynomial
        --spectrum-decay-param "${p}"
        --spectrum-rank 8192
        --subspace-iter 1
        --device-random-input
        --skip-form-u
        --compress-b-mode "${b_mode}"
        --compress-b-bits "${b_bits}"
        --compress-subspace-mode "${z_mode}"
        --compress-subspace-bits "${z_bits}"
        --summary-only
    )

    local -a b_error_args=()
    if [[ "${CHECK_B_ERROR}" == "1" && "${label}" != "none" ]]; then
        b_error_args+=(--check-b-error)
    fi

    echo "====================================="
    echo "CASE p=${p} method=${label} m=32768 n=8192 spectrum_rank=8192"
    echo "Date: $(date)"

    echo "Timing: p=${p} method=${label}"
    srun --mpi=pmix "$BIN" \
        "${base_args[@]}" \
        --repeat "${REPEAT}" \
        --no-check-error
    echo "-------------------------------------"

    echo "Final-error: p=${p} method=${label}"
    srun --mpi=pmix "$BIN" \
        "${base_args[@]}" \
        --repeat "${REPEAT}" \
        "${b_error_args[@]}"
}

for p in ${PVALS}; do
    run_case "${p}" "none" "none" 0 "none" 0
    run_case "${p}" "tq8" "tq" 8 "tq" 8
    run_case "${p}" "tq4" "tq" 4 "tq" 4
done

echo "Synthetic accuracy sweep complete."

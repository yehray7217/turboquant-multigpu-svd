#!/bin/bash
#SBATCH --job-name=v4_syn_scale
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=$HOME/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_synthetic_accuracy_scaling/output_logs/scaling_%j.out
#SBATCH --error=$HOME/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_synthetic_accuracy_scaling/error_logs/scaling_%j.err

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
RUN_DUPLICATE_32K="${RUN_DUPLICATE_32K:-0}"

cd "$PROJECT_ROOT"
make
cd "$EXP_DIR"

echo "Synthetic matrix-size scaling"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "REPEAT=${REPEAT} CHECK_B_ERROR=${CHECK_B_ERROR} RUN_DUPLICATE_32K=${RUN_DUPLICATE_32K}"
nvidia-smi -L
echo "====================================="

run_case() {
    local m="$1"
    local n="$2"
    local spectrum_rank="$3"
    local label="$4"
    local b_mode="$5"
    local b_bits="$6"
    local z_mode="$7"
    local z_bits="$8"

    local -a base_args=(
        --m "${m}"
        --n "${n}"
        --k 250
        --oversample 6
        --ngpus 16
        --gpus-per-rank 8
        --seed 1234
        --spectrum-decay-mode polynomial
        --spectrum-decay-param 0.6
        --spectrum-rank "${spectrum_rank}"
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
    echo "CASE size=${m}x${n} method=${label} p=0.6 spectrum_rank=${spectrum_rank}"
    echo "Date: $(date)"

    echo "Timing: size=${m}x${n} method=${label}"
    srun --mpi=pmix "$BIN" \
        "${base_args[@]}" \
        --repeat "${REPEAT}" \
        --no-check-error
    echo "-------------------------------------"

    echo "Final-error: size=${m}x${n} method=${label}"
    srun --mpi=pmix "$BIN" \
        "${base_args[@]}" \
        --repeat "${REPEAT}" \
        "${b_error_args[@]}"
}

run_size() {
    local m="$1"
    local n="$2"
    local spectrum_rank="$3"

    run_case "${m}" "${n}" "${spectrum_rank}" "none" "none" 0 "none" 0
    run_case "${m}" "${n}" "${spectrum_rank}" "tq8" "tq" 8 "tq" 8
    run_case "${m}" "${n}" "${spectrum_rank}" "tq4" "tq" 4 "tq" 4
}

run_size 16384 4096 4096

if [[ "${RUN_DUPLICATE_32K}" == "1" ]]; then
    run_size 32768 8192 8192
else
    echo "Skipping duplicate 32768x8192 p=0.6 point; reuse the Experiment A p=0.6 results."
    echo "Set RUN_DUPLICATE_32K=1 to rerun it in this script."
fi

echo "Synthetic scaling experiment complete."

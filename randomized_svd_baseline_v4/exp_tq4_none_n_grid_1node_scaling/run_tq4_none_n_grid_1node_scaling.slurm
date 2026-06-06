#!/bin/bash
#SBATCH --job-name=v4_tq4_1node_scale
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=tq4_none_n_grid_1node_scaling_%j.out
#SBATCH --error=tq4_none_n_grid_1node_scaling_%j.err

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

cd "${PROJECT_ROOT}/exp_tq4_none_n_grid_1node_scaling"

MATRIX_M="${MATRIX_M:-65536}"
N_START="${N_START:-16384}"
N_STOP="${N_STOP:-327680}"
N_STEP="${N_STEP:-16384}"
REPEAT="${REPEAT:-5}"
GPU_COUNTS="${GPU_COUNTS:-1 2 4}"
ERROR_MODE="${ERROR_MODE:-none}"
CHECK_B_ERROR="${CHECK_B_ERROR:-0}"

COMMON_ARGS=(
    --m "${MATRIX_M}"
    --k 250
    --oversample 6
    --seed 1234
    --spectrum-decay-mode polynomial
    --spectrum-decay-param 0.6
    --spectrum-rank 8192
    --subspace-iter 1
    --device-random-input
    --skip-form-u
    --summary-only
)

if [[ "${ERROR_MODE}" == "none" || "${ERROR_MODE}" == "b" ]]; then
    COMMON_ARGS+=(--no-check-error)
elif [[ "${ERROR_MODE}" != "final" ]]; then
    echo "ERROR_MODE must be one of: final, b, none" >&2
    exit 2
fi

if [[ "${ERROR_MODE}" == "b" || "${CHECK_B_ERROR}" == "1" ]]; then
    COMMON_ARGS+=(--check-b-error)
fi

run_case() {
    local label="$1"
    local n="$2"
    local gpu_count="$3"
    local b_mode="$4"
    local b_bits="$5"
    local z_mode="$6"
    local z_bits="$7"

    echo "====================================="
    echo "GRID_CASE method=${label} nodes=1 mpi_ranks=1 m=${MATRIX_M} n=${n} repeat=${REPEAT} ngpus=${gpu_count}"
    echo "Date: $(date)"

    srun --mpi=pmix --ntasks=1 "$BIN" \
        "${COMMON_ARGS[@]}" \
        --n "${n}" \
        --ngpus "${gpu_count}" \
        --gpus-per-rank "${gpu_count}" \
        --compress-b-mode "${b_mode}" \
        --compress-b-bits "${b_bits}" \
        --compress-subspace-mode "${z_mode}" \
        --compress-subspace-bits "${z_bits}" \
        --repeat "${REPEAT}"
}

echo "v4 1-node 1/2/4-GPU NONE/TQ4 n-grid"
echo "Host: $(hostname)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "MATRIX_M=${MATRIX_M} N_START=${N_START} N_STOP=${N_STOP} N_STEP=${N_STEP} REPEAT=${REPEAT} GPU_COUNTS=${GPU_COUNTS} ERROR_MODE=${ERROR_MODE} CHECK_B_ERROR=${CHECK_B_ERROR}"
nvidia-smi -L

for ((n = N_START; n <= N_STOP; n += N_STEP)); do
    for gpu_count in ${GPU_COUNTS}; do
        run_case "none" "${n}" "${gpu_count}" "none" 0 "none" 0
        run_case "TQ4" "${n}" "${gpu_count}" "tq" 4 "tq" 4
    done
done

echo "1-node scaling n-grid complete."

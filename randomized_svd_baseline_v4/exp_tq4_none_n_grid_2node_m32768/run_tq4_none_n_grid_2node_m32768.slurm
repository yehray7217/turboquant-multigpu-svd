#!/bin/bash
#SBATCH --job-name=v4_tq4_2node_grid_m32768
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=tq4_none_n_grid_2node_m32768_%j.out
#SBATCH --error=tq4_none_n_grid_2node_m32768_%j.err

set -euo pipefail

module load cuda/12.8
module load ucx/1.14.1
module load openmpi/5.0.2_ucx1.14.1_cuda12.3

unset OMPI_MCA_opal_cuda_support || true
unset OMPI_MCA_btl_smcuda_use_cuda_ipc || true

PROJECT_ROOT="/home/terryyang1/turboquant-multigpu-svd/randomized_svd_baseline_v4"
BIN="${PROJECT_ROOT}/.build/randomized_svd_multigpu_v4"

cd "$PROJECT_ROOT"
make

cd "${PROJECT_ROOT}/exp_tq4_none_n_grid_2node_m32768"

MATRIX_M="${MATRIX_M:-32768}"
N_START="${N_START:-16384}"
N_STOP="${N_STOP:-655360}"
N_STEP="${N_STEP:-16384}"
REPEAT="${REPEAT:-5}"
NGPUS="${NGPUS:-16}"
GPUS_PER_RANK="${GPUS_PER_RANK:-8}"
ERROR_MODE="${ERROR_MODE:-none}"
CHECK_B_ERROR="${CHECK_B_ERROR:-0}"

COMMON_ARGS=(
    --m "${MATRIX_M}"
    --k 250
    --oversample 6
    --ngpus "${NGPUS}"
    --gpus-per-rank "${GPUS_PER_RANK}"
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
    local b_mode="$3"
    local b_bits="$4"
    local z_mode="$5"
    local z_bits="$6"

    echo "====================================="
    echo "GRID_CASE method=${label} nodes=2 mpi_ranks=2 m=${MATRIX_M} n=${n} repeat=${REPEAT} ngpus=${NGPUS}"
    echo "Date: $(date)"

    srun --mpi=pmix "$BIN" \
        "${COMMON_ARGS[@]}" \
        --n "${n}" \
        --compress-b-mode "${b_mode}" \
        --compress-b-bits "${b_bits}" \
        --compress-subspace-mode "${z_mode}" \
        --compress-subspace-bits "${z_bits}" \
        --repeat "${REPEAT}"
}

echo "v4 2-node NONE/TQ4 n-grid (fixed m=32768, communication payload scaling)"
echo "Host: $(hostname)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "MATRIX_M=${MATRIX_M} N_START=${N_START} N_STOP=${N_STOP} N_STEP=${N_STEP} REPEAT=${REPEAT} ERROR_MODE=${ERROR_MODE} CHECK_B_ERROR=${CHECK_B_ERROR}"
srun --ntasks=2 --ntasks-per-node=1 nvidia-smi -L

for ((n = N_START; n <= N_STOP; n += N_STEP)); do
    run_case "none" "${n}" "none" 0 "none" 0
    run_case "TQ4" "${n}" "tq" 4 "tq" 4
done

echo "2-node m=32768 n-grid complete."

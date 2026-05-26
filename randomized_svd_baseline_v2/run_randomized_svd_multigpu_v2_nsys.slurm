#!/bin/bash
#SBATCH --job-name=rand_svd_v2_nsys
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --time=00:30:00
#SBATCH --output=randomized_svd_multigpu_v2_nsys_%j.out
#SBATCH --error=randomized_svd_multigpu_v2_nsys_%j.err

set -euo pipefail

module load cuda/12.8

PROJECT_ROOT=/home/yehray7217/turboquant-multigpu-svd/randomized_svd_baseline_v2
cd "$PROJECT_ROOT"

make

EXTRA_ARGS=()
if [[ "${DEVICE_RANDOM_INPUT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--device-random-input)
fi
if [[ "${SKIP_FORM_U:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--skip-form-u)
fi

echo "Starting Nsight Systems profile"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "DEVICE_RANDOM_INPUT=${DEVICE_RANDOM_INPUT:-0}"
echo "SKIP_FORM_U=${SKIP_FORM_U:-0}"
nvidia-smi -L
echo "====================================="

run_profile() {
    local LABEL="$1"
    local MODE="$2"
    local BITS="$3"
    local QJL_DIM="$4"
    local QJL_ALPHA="$5"
    local REPORT="nsys_rand_svd_v2_${LABEL}_${SLURM_JOB_ID}"

    echo "Report: ${REPORT}.nsys-rep"
    echo "Running profile: mode=${MODE} bits=${BITS} qjl_dim=${QJL_DIM} qjl_alpha=${QJL_ALPHA}"
    nsys profile \
        --force-overwrite=true \
        --trace=cuda,nvtx,cublas,cusolver,osrt \
        --sample=none \
        --stats=true \
        --output "$REPORT" \
        ./randomized_svd_multigpu_v2 \
            --m 32768 \
            --n 8192 \
            --k 256 \
            --oversample 64 \
            --ngpus 2 \
            --compress-b-mode "$MODE" \
            --compress-b-bits "$BITS" \
            --qjl-dim "$QJL_DIM" \
            --qjl-alpha "$QJL_ALPHA" \
            --no-check-error \
            --no-check-b-error \
            "${EXTRA_ARGS[@]}"
}

run_profile "tq4" "tq" "4" "0" "0.0"
run_profile "tqqjl4" "tq-qjl" "4" "64" "0.01"

echo "Nsight Systems profile complete!"

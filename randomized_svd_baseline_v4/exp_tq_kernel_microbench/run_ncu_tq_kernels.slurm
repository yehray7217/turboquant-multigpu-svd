#!/bin/bash
#SBATCH --job-name=tq_kernel_ncu
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=tq_kernel_ncu_%j.out
#SBATCH --error=tq_kernel_ncu_%j.err

set -euo pipefail

module load cuda/12.8

cd "$HOME/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_tq_kernel_microbench"

make

if ! command -v ncu >/dev/null 2>&1; then
    echo "ERROR: ncu not found after module load cuda/12.8"
    exit 1
fi

echo "TQ kernel Nsight Compute profile"
echo "Host: $(hostname)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
nvidia-smi -L
ncu --version
echo "====================================="

NCU_OUT_DIR="ncu_reports"
mkdir -p "${NCU_OUT_DIR}"

COMMON_ARGS=(
    --rows 256
    --cols 16384
    --bits 4
    --warmup 0
    --repeat 1
    --seed 1234
)

profile_kernel() {
    local label="$1"
    local kernel_regex="$2"
    local report="${NCU_OUT_DIR}/${label}_${SLURM_JOB_ID}"

    echo "Profiling ${label}"
    ncu \
        --target-processes all \
        --kernel-name "regex:${kernel_regex}" \
        --launch-count 1 \
        --set speedOfLight \
        --section LaunchStats \
        --section Occupancy \
        --section MemoryWorkloadAnalysis \
        --section WarpStateStats \
        --export "${report}" \
        --force-overwrite \
        ./.build/tq_kernel_microbench "${COMMON_ARGS[@]}"
    echo "Report: ${report}.ncu-rep"
    echo "-------------------------------------"
}

profile_kernel \
    "norm" \
    ".*column_norms_kernel.*"

profile_kernel \
    "fused_fwht256" \
    ".*column_tq_normalize_sign_fwht256_kernel.*"

profile_kernel \
    "quantize_bitpack" \
    ".*column_tq_lloyd_quantize_kernel.*"

profile_kernel \
    "decode_add" \
    ".*column_tq4?_lloyd_dequantize_fwht256_apply_add_kernel.*"

echo "Nsight Compute profiling complete."

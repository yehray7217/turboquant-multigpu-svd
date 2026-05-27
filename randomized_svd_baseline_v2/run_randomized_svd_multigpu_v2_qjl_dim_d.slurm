#!/bin/bash
#SBATCH --job-name=rand_svd_v2_qjld
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --time=00:30:00
#SBATCH --output=randomized_svd_multigpu_v2_qjl_dim_d_%j.out
#SBATCH --error=randomized_svd_multigpu_v2_qjl_dim_d_%j.err

set -euo pipefail

module load cuda/12.8

PROJECT_ROOT=$HOME/turboquant-multigpu-svd/randomized_svd_baseline_v2
cd "$PROJECT_ROOT"

make

M=32768
N=8192
K=256
OVERSAMPLE=64
QJL_DIM=$((K + OVERSAMPLE))

COMMON_ARGS=(
    --m "$M"
    --n "$N"
    --k "$K"
    --oversample "$OVERSAMPLE"
    --ngpus 2
    --seed 1234
    --device-random-input
    --skip-form-u
    --repeat 1
    --repeat-print-every 1
    --no-check-error
)

echo "Starting Randomized Multi-GPU SVD v2 QJL dim=d sweep"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "QJL_DIM=${QJL_DIM} (= k + oversample = column dimension l)"
nvidia-smi -L
echo "====================================="

echo "Running B-error baseline: tq 4-bit"
./randomized_svd_multigpu_v2 \
    "${COMMON_ARGS[@]}" \
    --compress-b-mode tq \
    --compress-b-bits 4 \
    --qjl-dim 0 \
    --qjl-alpha 0.0
echo "-------------------------------------"

for alpha in -1.0 -0.1 -0.01 -0.001 0.0 0.001 0.01 0.1 1.0; do
    echo "Running dim=d sweep: tq-qjl 4-bit dim=${QJL_DIM} alpha=${alpha}"
    ./randomized_svd_multigpu_v2 \
        "${COMMON_ARGS[@]}" \
        --compress-b-mode tq-qjl \
        --compress-b-bits 4 \
        --qjl-dim "$QJL_DIM" \
        --qjl-alpha "$alpha"
    echo "-------------------------------------"
done

echo "QJL dim=d sweep complete!"

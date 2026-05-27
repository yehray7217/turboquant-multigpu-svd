#!/bin/bash
#SBATCH --job-name=rand_svd_v2_tqfocus
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --time=00:30:00
#SBATCH --output=randomized_svd_multigpu_v2_tq_focus_%j.out
#SBATCH --error=randomized_svd_multigpu_v2_tq_focus_%j.err

set -euo pipefail

module load cuda/12.8

PROJECT_ROOT=$HOME/turboquant-multigpu-svd/randomized_svd_baseline_v2
cd "$PROJECT_ROOT"

make

COMMON_ARGS=(
    --m 32768
    --n 8192
    --k 256
    --oversample 64
    --ngpus 2
    --seed 1234
    --device-random-input
    --skip-form-u
    --repeat 200
    --repeat-print-every 25
    --no-check-error
    --no-check-b-error
)

echo "Starting Randomized Multi-GPU SVD v2 TQ-focused sweep"
echo "Date: $(date)"
echo "Host: $(hostname)"
nvidia-smi -L
echo "====================================="

echo "Running TQ focus: none"
./randomized_svd_multigpu_v2 \
    "${COMMON_ARGS[@]}" \
    --compress-b-mode none \
    --compress-b-bits 0
echo "-------------------------------------"

echo "Running TQ focus: tq 4-bit"
./randomized_svd_multigpu_v2 \
    "${COMMON_ARGS[@]}" \
    --compress-b-mode tq \
    --compress-b-bits 4 \
    --qjl-dim 0 \
    --qjl-alpha 0.0
echo "-------------------------------------"

echo "Running TQ focus: tq 2-bit"
./randomized_svd_multigpu_v2 \
    "${COMMON_ARGS[@]}" \
    --compress-b-mode tq \
    --compress-b-bits 2 \
    --qjl-dim 0 \
    --qjl-alpha 0.0
echo "-------------------------------------"

echo "Running TQ focus: tq-qjl 4-bit"
./randomized_svd_multigpu_v2 \
    "${COMMON_ARGS[@]}" \
    --compress-b-mode tq-qjl \
    --compress-b-bits 4 \
    --qjl-dim 64 \
    --qjl-alpha 0.01
echo "-------------------------------------"

echo "Running TQ focus: tq-qjl 2-bit"
./randomized_svd_multigpu_v2 \
    "${COMMON_ARGS[@]}" \
    --compress-b-mode tq-qjl \
    --compress-b-bits 2 \
    --qjl-dim 64 \
    --qjl-alpha 0.01
echo "-------------------------------------"

echo "TQ-focused sweep complete!"

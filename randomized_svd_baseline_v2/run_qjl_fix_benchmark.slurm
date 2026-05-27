#!/bin/bash
#SBATCH --job-name=qjl_fix_bench
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --output=qjl_fix_benchmark_%j.out
#SBATCH --error=qjl_fix_benchmark_%j.err

# Purpose: Benchmark TQ+QJL timing before vs. after pre-allocated scratch buffer fix.
# Target config: 1 node, 8 GPUs, same matrix as RECORD.csv baseline.
# Expected: TQ+QJL should drop from ~161ms to ~50-60ms after the fix.

set -euo pipefail

module load cuda/12.8

PROJECT_ROOT=/home/terryyang1/turboquant-multigpu-svd/randomized_svd_baseline_v2
cd "$PROJECT_ROOT"

make

echo "====================================="
echo "QJL Fix Benchmark"
echo "Date: $(date)"
echo "Host: $(hostname)"
nvidia-smi -L
echo "====================================="

COMMON_ARGS=(
    --m 32768
    --n 8192
    --k 256
    --oversample 64
    --ngpus 8
    --seed 1234
    --device-random-input
    --skip-form-u
    --repeat 100
    --repeat-print-every 25
    --no-check-error
    --no-check-b-error
)

echo "--- none (baseline) ---"
./randomized_svd_multigpu_v2 \
    "${COMMON_ARGS[@]}" \
    --compress-b-mode none \
    --compress-b-bits 0
echo "-------------------------------------"

echo "--- TQ 4-bit ---"
./randomized_svd_multigpu_v2 \
    "${COMMON_ARGS[@]}" \
    --compress-b-mode tq \
    --compress-b-bits 4 \
    --qjl-dim 0 \
    --qjl-alpha 0.0
echo "-------------------------------------"

echo "--- TQ 2-bit ---"
./randomized_svd_multigpu_v2 \
    "${COMMON_ARGS[@]}" \
    --compress-b-mode tq \
    --compress-b-bits 2 \
    --qjl-dim 0 \
    --qjl-alpha 0.0
echo "-------------------------------------"

echo "--- TQ+QJL 4-bit (qjl_dim=64, alpha=0.01) ---"
./randomized_svd_multigpu_v2 \
    "${COMMON_ARGS[@]}" \
    --compress-b-mode tq-qjl \
    --compress-b-bits 4 \
    --qjl-dim 64 \
    --qjl-alpha 0.01
echo "-------------------------------------"

echo "--- TQ+QJL 2-bit (qjl_dim=64, alpha=0.01) ---"
./randomized_svd_multigpu_v2 \
    "${COMMON_ARGS[@]}" \
    --compress-b-mode tq-qjl \
    --compress-b-bits 2 \
    --qjl-dim 64 \
    --qjl-alpha 0.01
echo "-------------------------------------"

echo "QJL Fix Benchmark complete!"

#!/bin/bash
#SBATCH --job-name=fp16_baseline
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --output=fp16_baseline_%j.out
#SBATCH --error=fp16_baseline_%j.err

# Purpose: Measure FP16 B-reduction baseline (2x compression, near-lossless ~0.001 error)
# as a comparison point vs TQ 4-bit (5x, 0.171 error) and TQ 2-bit (10x, 0.948 error).
# Target config: 1 node, 8 GPUs, same matrix as RECORD.csv baseline.
# Runs all 5 modes back-to-back to keep matrix/random/setup costs comparable.

set -euo pipefail

module load cuda/12.8

PROJECT_ROOT=/home/terryyang1/turboquant-multigpu-svd/randomized_svd_baseline_v2
cd "$PROJECT_ROOT"

make

echo "====================================="
echo "FP16 Baseline Benchmark"
echo "Date: $(date)"
echo "Host: $(hostname)"
nvidia-smi -L
echo "====================================="

# Two pass strategy:
#   Pass 1: timing-only (100 repeats, --no-check-b-error) for warm pipeline averages.
#   Pass 2: B-error only on FP16 (5 repeats, B-error check on) for reduce_B_relative_error.
# Avoids mixing B-error host copies into timing numbers (lesson from job 928995).

TIMING_ARGS=(
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

echo "========== PASS 1: TIMING (100 repeats) =========="

echo "--- none (baseline) ---"
./randomized_svd_multigpu_v2 \
    "${TIMING_ARGS[@]}" \
    --compress-b-mode none \
    --compress-b-bits 0
echo "-------------------------------------"

echo "--- FP16 (NEW: 2x near-lossless) ---"
./randomized_svd_multigpu_v2 \
    "${TIMING_ARGS[@]}" \
    --compress-b-mode fp16 \
    --compress-b-bits 0
echo "-------------------------------------"

echo "--- TQ 4-bit ---"
./randomized_svd_multigpu_v2 \
    "${TIMING_ARGS[@]}" \
    --compress-b-mode tq \
    --compress-b-bits 4 \
    --qjl-dim 0 \
    --qjl-alpha 0.0
echo "-------------------------------------"

echo "--- TQ 2-bit ---"
./randomized_svd_multigpu_v2 \
    "${TIMING_ARGS[@]}" \
    --compress-b-mode tq \
    --compress-b-bits 2 \
    --qjl-dim 0 \
    --qjl-alpha 0.0
echo "-------------------------------------"

echo "--- TQ+QJL 4-bit (dead-end reference) ---"
./randomized_svd_multigpu_v2 \
    "${TIMING_ARGS[@]}" \
    --compress-b-mode tq-qjl \
    --compress-b-bits 4 \
    --qjl-dim 64 \
    --qjl-alpha 0.01
echo "-------------------------------------"

BERROR_ARGS=(
    --m 32768
    --n 8192
    --k 256
    --oversample 64
    --ngpus 8
    --seed 1234
    --device-random-input
    --skip-form-u
    --repeat 5
    --repeat-print-every 5
    --no-check-error
    # NOTE: --no-check-b-error intentionally OMITTED
)

echo "========== PASS 2: B-ERROR (5 repeats, timing inflated) =========="
echo "--- FP16 B-error check ---"
./randomized_svd_multigpu_v2 \
    "${BERROR_ARGS[@]}" \
    --compress-b-mode fp16 \
    --compress-b-bits 0
echo "-------------------------------------"

echo "FP16 Baseline Benchmark complete!"

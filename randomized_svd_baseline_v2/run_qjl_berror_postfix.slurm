#!/bin/bash
#SBATCH --job-name=qjl_berror_pf
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=00:15:00
#SBATCH --output=qjl_berror_postfix_%j.out
#SBATCH --error=qjl_berror_postfix_%j.err

# Purpose: Measure B-error for TQ vs TQ+QJL after the pre-allocation scratch fix.
# Confirms whether QJL error behavior changed (expected: still worsens vs TQ alone).

set -euo pipefail

module load cuda/12.8

PROJECT_ROOT=/home/terryyang1/turboquant-multigpu-svd/randomized_svd_baseline_v2
cd "$PROJECT_ROOT"

make

echo "====================================="
echo "QJL B-Error Check (Post-Fix)"
echo "Date: $(date)"
echo "Host: $(hostname)"
nvidia-smi -L
echo "====================================="

# NOTE: B-error check copies B_i back to host — slow.
# Use small repeat count (5) to keep runtime short.
COMMON_ARGS=(
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
    # NOTE: --no-check-b-error is intentionally OMITTED → B-error will be computed
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

echo "--- TQ+QJL 4-bit (qjl_dim=64, alpha=0.01) ---"
./randomized_svd_multigpu_v2 \
    "${COMMON_ARGS[@]}" \
    --compress-b-mode tq-qjl \
    --compress-b-bits 4 \
    --qjl-dim 64 \
    --qjl-alpha 0.01
echo "-------------------------------------"

echo "--- TQ+QJL 4-bit (qjl_dim=64, alpha=0.0) --- [alpha=0 == pure TQ, sanity check]"
./randomized_svd_multigpu_v2 \
    "${COMMON_ARGS[@]}" \
    --compress-b-mode tq-qjl \
    --compress-b-bits 4 \
    --qjl-dim 64 \
    --qjl-alpha 0.0
echo "-------------------------------------"

echo "QJL B-Error Post-Fix check complete!"

#!/bin/bash
#SBATCH --job-name=rand_svd_v2_qjlag
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --time=00:30:00
#SBATCH --output=randomized_svd_multigpu_v2_qjl_alpha_grid_%j.out
#SBATCH --error=randomized_svd_multigpu_v2_qjl_alpha_grid_%j.err

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
    --repeat 1
    --repeat-print-every 1
    --no-check-error
)

echo "Starting Randomized Multi-GPU SVD v2 QJL broad alpha grid"
echo "Date: $(date)"
echo "Host: $(hostname)"
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

for dim in 16 32; do
    for alpha in -1.0 -0.5 -0.25 -0.1 -0.05 -0.01 0.0 0.01 0.05 0.1 0.25 0.5 1.0; do
        echo "Running broad alpha grid: tq-qjl 4-bit dim=${dim} alpha=${alpha}"
        ./randomized_svd_multigpu_v2 \
            "${COMMON_ARGS[@]}" \
            --compress-b-mode tq-qjl \
            --compress-b-bits 4 \
            --qjl-dim "${dim}" \
            --qjl-alpha "${alpha}"
        echo "-------------------------------------"
    done
done

echo "QJL broad alpha grid complete!"

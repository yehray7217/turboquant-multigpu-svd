#!/bin/bash
#SBATCH --job-name=rand_svd_v2_tqbits
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --time=00:30:00
#SBATCH --output=randomized_svd_multigpu_v2_tq_bit_curve_%j.out
#SBATCH --error=randomized_svd_multigpu_v2_tq_bit_curve_%j.err

set -euo pipefail

module load cuda/12.8

PROJECT_ROOT=/home/yehray7217/turboquant-multigpu-svd/randomized_svd_baseline_v2
cd "$PROJECT_ROOT"

make

BASE_ARGS=(
    --m 32768
    --n 8192
    --k 256
    --oversample 64
    --ngpus 2
    --seed 1234
    --device-random-input
    --skip-form-u
)

echo "Starting Randomized Multi-GPU SVD v2 TQ bit/time/error curve"
echo "Date: $(date)"
echo "Host: $(hostname)"
nvidia-smi -L
echo "====================================="

TIME_ARGS=(
    "${BASE_ARGS[@]}"
    --repeat 100
    --repeat-print-every 25
    --no-check-error
    --no-check-b-error
)

echo "Timing: none"
./randomized_svd_multigpu_v2 \
    "${TIME_ARGS[@]}" \
    --compress-b-mode none \
    --compress-b-bits 0
echo "-------------------------------------"

for bits in 4 2; do
    echo "Timing: tq ${bits}-bit"
    ./randomized_svd_multigpu_v2 \
        "${TIME_ARGS[@]}" \
        --compress-b-mode tq \
        --compress-b-bits "${bits}" \
        --qjl-dim 0 \
        --qjl-alpha 0.0
    echo "-------------------------------------"
done

ERROR_ARGS=(
    "${BASE_ARGS[@]}"
    --repeat 1
    --repeat-print-every 1
    --no-check-error
)

for bits in 4 2; do
    echo "B-error: tq ${bits}-bit"
    ./randomized_svd_multigpu_v2 \
        "${ERROR_ARGS[@]}" \
        --compress-b-mode tq \
        --compress-b-bits "${bits}" \
        --qjl-dim 0 \
        --qjl-alpha 0.0
    echo "-------------------------------------"
done

echo "TQ bit/time/error curve complete!"

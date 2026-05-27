#!/bin/bash
#SBATCH --job-name=rand_svd_v2_scale
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=randomized_svd_multigpu_v2_scaling_1_2_4_8_%j.out
#SBATCH --error=randomized_svd_multigpu_v2_scaling_1_2_4_8_%j.err

set -euo pipefail

module load cuda/12.8

PROJECT_ROOT=$HOME/turboquant-multigpu-svd/randomized_svd_baseline_v2
cd "$PROJECT_ROOT"

make

echo "Starting Randomized Multi-GPU SVD v2 1/2/4/8 GPU scaling"
echo "Date: $(date)"
echo "Host: $(hostname)"
nvidia-smi -L
echo "====================================="

for NGPUS in 1 2 4 8; do
    BASE_ARGS=(
        --m 32768
        --n 8192
        --k 256
        --oversample 64
        --ngpus "${NGPUS}"
        --seed 1234
        --device-random-input
        --skip-form-u
    )

    TIME_ARGS=(
        "${BASE_ARGS[@]}"
        --repeat 50
        --repeat-print-every 25
        --no-check-error
        --no-check-b-error
    )

    echo "Scaling timing: ngpus=${NGPUS} mode=none"
    ./randomized_svd_multigpu_v2 \
        "${TIME_ARGS[@]}" \
        --compress-b-mode none \
        --compress-b-bits 0
    echo "-------------------------------------"

    echo "Scaling timing: ngpus=${NGPUS} mode=tq bits=4"
    ./randomized_svd_multigpu_v2 \
        "${TIME_ARGS[@]}" \
        --compress-b-mode tq \
        --compress-b-bits 4 \
        --qjl-dim 0 \
        --qjl-alpha 0.0
    echo "-------------------------------------"

    echo "Scaling timing: ngpus=${NGPUS} mode=tq bits=2"
    ./randomized_svd_multigpu_v2 \
        "${TIME_ARGS[@]}" \
        --compress-b-mode tq \
        --compress-b-bits 2 \
        --qjl-dim 0 \
        --qjl-alpha 0.0
    echo "-------------------------------------"
done

for NGPUS in 1 2 4 8; do
    ERROR_ARGS=(
        --m 32768
        --n 8192
        --k 256
        --oversample 64
        --ngpus "${NGPUS}"
        --seed 1234
        --device-random-input
        --skip-form-u
        --repeat 1
        --repeat-print-every 1
        --no-check-error
    )

    echo "Scaling B-error: ngpus=${NGPUS} mode=tq bits=4"
    ./randomized_svd_multigpu_v2 \
        "${ERROR_ARGS[@]}" \
        --compress-b-mode tq \
        --compress-b-bits 4 \
        --qjl-dim 0 \
        --qjl-alpha 0.0
    echo "-------------------------------------"

    echo "Scaling B-error: ngpus=${NGPUS} mode=tq bits=2"
    ./randomized_svd_multigpu_v2 \
        "${ERROR_ARGS[@]}" \
        --compress-b-mode tq \
        --compress-b-bits 2 \
        --qjl-dim 0 \
        --qjl-alpha 0.0
    echo "-------------------------------------"
done

echo "Randomized Multi-GPU SVD v2 1/2/4/8 GPU scaling complete!"

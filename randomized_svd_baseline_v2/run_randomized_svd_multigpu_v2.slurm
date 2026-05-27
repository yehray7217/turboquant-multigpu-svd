#!/bin/bash
#SBATCH --job-name=rand_svd_v2_2gpu
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --time=00:30:00
#SBATCH --output=randomized_svd_multigpu_v2_%j.out
#SBATCH --error=randomized_svd_multigpu_v2_%j.err

set -euo pipefail

module load cuda/12.8

PROJECT_ROOT=$HOME/turboquant-multigpu-svd/randomized_svd_baseline_v2
cd "$PROJECT_ROOT"

make

# Test configurations: M N K OVERSAMPLE NGPUS
CONFIGS=(
    "4096 2048 64 16 2"
    "8192 4096 128 32 2"
)

# Compression configurations: MODE BITS QJL_DIM QJL_ALPHA
COMPRESS_CONFIGS=(
    "none 0 0 0.0"
    "lowbit 8 0 0.0"
    "lowbit 4 0 0.0"
    "lowbit 2 0 0.0"
    "tq 8 0 0.0"
    "tq 4 0 0.0"
    "tq 2 0 0.0"
    "tq-qjl 8 64 0.01"
    "tq-qjl 4 64 0.01"
    "tq-qjl 2 64 0.01"
)

echo "Starting Randomized Multi-GPU SVD v2 Benchmark"
echo "Date: $(date)"
echo "Host: $(hostname)"
nvidia-smi -L
echo "====================================="

for config in "${CONFIGS[@]}"
do
    read -r M N K OVERSAMPLE NGPUS <<< "$config"
    for compress_config in "${COMPRESS_CONFIGS[@]}"
    do
        read -r MODE BITS QJL_DIM QJL_ALPHA <<< "$compress_config"
        echo "Running: M=$M N=$N K=$K OVERSAMPLE=$OVERSAMPLE NGPUS=$NGPUS COMPRESS_B_MODE=$MODE COMPRESS_B_BITS=$BITS QJL_DIM=$QJL_DIM QJL_ALPHA=$QJL_ALPHA"
        ./randomized_svd_multigpu_v2 \
            --m "$M" \
            --n "$N" \
            --k "$K" \
            --oversample "$OVERSAMPLE" \
            --ngpus "$NGPUS" \
            --compress-b-mode "$MODE" \
            --compress-b-bits "$BITS" \
            --qjl-dim "$QJL_DIM" \
            --qjl-alpha "$QJL_ALPHA"
        echo "-------------------------------------"
    done
done

echo "Benchmark Complete!"

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

PROJECT_ROOT=/home/yehray7217/turboquant-multigpu-svd/randomized_svd_baseline_v2
cd "$PROJECT_ROOT"

make

# Test configurations: M N K OVERSAMPLE NGPUS
CONFIGS=(
    "4096 2048 64 16 2"
    "8192 4096 128 32 2"
)

echo "Starting Randomized Multi-GPU SVD v2 Benchmark"
echo "Date: $(date)"
echo "Host: $(hostname)"
nvidia-smi -L
echo "====================================="

for config in "${CONFIGS[@]}"
do
    read -r M N K OVERSAMPLE NGPUS <<< "$config"
    echo "Running: M=$M N=$N K=$K OVERSAMPLE=$OVERSAMPLE NGPUS=$NGPUS"
    ./randomized_svd_multigpu_v2 \
        --m "$M" \
        --n "$N" \
        --k "$K" \
        --oversample "$OVERSAMPLE" \
        --ngpus "$NGPUS"
    echo "-------------------------------------"
done

echo "Benchmark Complete!"

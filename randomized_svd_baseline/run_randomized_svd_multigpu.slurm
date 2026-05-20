#!/bin/bash
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=00:30:00

module load cuda/12.8

# Test configurations: M N K OVERSAMPLE NGPUS
CONFIGS=(
    "4096 2048 64 16 8"
    "8192 4096 128 32 8"
)

echo "Starting Randomized Multi-GPU SVD Benchmark"
echo "Date: $(date)"
echo "====================================="

for config in "${CONFIGS[@]}"
do
    read -r M N K OVERSAMPLE NGPUS <<< "$config"
    echo "Running: M=$M N=$N K=$K OVERSAMPLE=$OVERSAMPLE NGPUS=$NGPUS"
    ./randomized_svd_multigpu --m $M --n $N --k $K --oversample $OVERSAMPLE --ngpus $NGPUS
    echo "-------------------------------------"
done

echo "Benchmark Complete!"

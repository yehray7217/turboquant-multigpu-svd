#!/bin/bash
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00             

module load cuda/12.8

SIZES=(1024 2048 4096 8192 16384)

echo "Starting cuSOLVER Benchmark Sweep"
echo "Date: $(date)"
echo "====================================="

for size in "${SIZES[@]}"
do
    echo "Running SVD for Matrix: $size x $size"
    ./cusolver -M $size -N $size
    echo "-------------------------------------"
done

echo "Benchmark Complete!"
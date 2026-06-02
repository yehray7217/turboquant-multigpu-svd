#!/bin/bash
#SBATCH --job-name=qjl_ip_probe
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=output_logs/qjl_inner_product_probe_%j.out
#SBATCH --error=error_logs/qjl_inner_product_probe_%j.err

set -euo pipefail

module load cuda/12.8

cd "$HOME/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_qjl_inner_product"
make

echo "QJL inner-product sweep"
date
nvidia-smi -L

for sketch_dim in 64 128 256 512 1024 2048 4096; do
    ./.build/qjl_inner_product_probe \
        --m 4096 \
        --n 1024 \
        --l 256 \
        --sketch-dim "${sketch_dim}" \
        --seed 1234
done

#!/bin/bash
#SBATCH --job-name=qjl_ip_feas
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=output_logs/qjl_inner_product_blockwise_feasibility_%j.out
#SBATCH --error=error_logs/qjl_inner_product_blockwise_feasibility_%j.err

set -euo pipefail

module load cuda/12.8

cd "$HOME/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_qjl_inner_product"
make

echo "Block-wise QJL feasibility sweep"
date
nvidia-smi -L

for config in \
    "256 512" \
    "256 1024" \
    "128 1024" \
    "128 2048" \
    "64 2048"; do
    read -r block_rows sketch_dim <<< "${config}"
    ./.build/qjl_inner_product_probe \
        --m 4096 \
        --n 1024 \
        --l 256 \
        --block-rows "${block_rows}" \
        --sketch-dim "${sketch_dim}" \
        --seed 1234
done

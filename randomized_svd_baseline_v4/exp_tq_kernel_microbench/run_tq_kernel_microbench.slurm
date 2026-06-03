#!/bin/bash
#SBATCH --job-name=tq_kernel_micro
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=tq_kernel_microbench_%j.out
#SBATCH --error=tq_kernel_microbench_%j.err

set -euo pipefail

module load cuda/12.8

cd "$HOME/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_tq_kernel_microbench"

make

echo "TQ kernel microbenchmark"
echo "Host: $(hostname)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
nvidia-smi -L
echo "====================================="

for bits in 4 2 1; do
    for cols in 16384 32768; do
        ./.build/tq_kernel_microbench \
            --rows 256 \
            --cols "${cols}" \
            --bits "${bits}" \
            --warmup 20 \
            --repeat 200 \
            --seed 1234

        echo "====================================="
    done
done

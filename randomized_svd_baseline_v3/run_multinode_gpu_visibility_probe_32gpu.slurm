#!/bin/bash
#SBATCH --job-name=rand_svd_mn_probe32
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=00:05:00
#SBATCH --output=multinode_gpu_visibility_probe_32gpu_%j.out
#SBATCH --error=multinode_gpu_visibility_probe_32gpu_%j.err

set -euo pipefail

module load cuda/12.8

PROJECT_ROOT=$HOME/turboquant-multigpu-svd/randomized_svd_baseline_v3
cd "$PROJECT_ROOT"

echo "Starting multi-node GPU visibility probe, requested 32 GPUs"
echo "Date: $(date)"
echo "Submit host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NNODES=${SLURM_NNODES}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "====================================="

srun --mpi=pmix bash -lc '
    echo "rank=${SLURM_PROCID} local_rank=${SLURM_LOCALID} host=$(hostname)"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    nvidia-smi -L
    echo "-------------------------------------"
'

echo "Multi-node GPU visibility probe complete!"

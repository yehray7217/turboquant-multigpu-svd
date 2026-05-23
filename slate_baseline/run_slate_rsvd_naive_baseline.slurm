#!/bin/bash
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --job-name=slate-rsvd-naive
#SBATCH --output=slate-rsvd-naive-%j.out

set -eo pipefail

cd /home/rax10101010/hpc_final/turboquant-multigpu-svd/slate_baseline

# Load CUDA/MPI toolchain and local SLATE path (if already installed).
source ./env_taiwania2.sh

# Auto-install local SLATE only when missing.
if ! pkg-config --exists slate; then
    ./install_slate_local.sh
    source ./env_taiwania2.sh
fi

make -j

# 1 node x 2 ranks per node = 2 total ranks.
# Sweep multiple matrix sizes for baseline characterization.
CONFIGS=(
    "16384 2048 64 16"
    "32768 4096 128 32"
    "65536 4096 128 32"
    "131072 8192 256 64"
)

echo "Starting SLATE RSVD size sweep"
echo "Date: $(date)"
echo "====================================="

for config in "${CONFIGS[@]}"; do
    read -r M N K OVERSAMPLE <<< "$config"
    echo "Running: M=$M N=$N K=$K OVERSAMPLE=$OVERSAMPLE"
    srun --mpi=pmix ./slate_rsvd_naive_baseline \
        --m "$M" \
        --n "$N" \
        --k "$K" \
        --oversample "$OVERSAMPLE" \
        --seed 12345 \
        --expected-mpi-size 2 \
        --grid-p 1 \
        --grid-q 2
    echo "-------------------------------------"
done

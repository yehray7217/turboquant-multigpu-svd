#!/bin/bash
#SBATCH --job-name=v3_qjl_postfix
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=v3_qjl_postfix_%j.out
#SBATCH --error=v3_qjl_postfix_%j.err

# Loop 4 A1: Verify Loop 1's QJL pre-allocation fix on v3 multi-node (2 nodes / 16 GPU).
# v2 saw 57% speedup (job 928869). v3 code change went in same Loop 1 commit but was
# never benchmarked on multi-node MPI hardware.
#
# Pre-fix RECORD.csv numbers to beat:
#   TQ+QJL 4-bit, 32k x 8k : 205.19 ms  → expect ~80 ms post-fix
#   TQ+QJL 4-bit, 64k x 16k: 379.72 ms  → expect ~150 ms post-fix
#
# Also re-runs none / TQ 4-bit as sanity baselines (should match existing RECORD ±5%).

set -euo pipefail

module load cuda/12.8
module load ucx/1.14.1
module load openmpi/5.0.2_ucx1.14.1_cuda12.3

unset OMPI_MCA_opal_cuda_support || true
unset OMPI_MCA_btl_smcuda_use_cuda_ipc || true

PROJECT_ROOT=$HOME/turboquant-multigpu-svd/randomized_svd_baseline_v3
cd "$PROJECT_ROOT"

make

echo "====================================="
echo "v3 QJL Post-Fix Verification (Loop 4 A1)"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
nvidia-smi -L
echo "====================================="

TIME_ARGS_COMMON=(
    --k 256
    --oversample 64
    --ngpus 16
    --gpus-per-rank 8
    --seed 1234
    --device-random-input
    --skip-form-u
    --repeat 100
    --repeat-print-every 25
    --no-check-error
    --no-check-b-error
)

run_sweep() {
    local m=$1
    local n=$2
    echo "========== MATRIX m=${m} n=${n} =========="

    echo "--- none ---"
    srun --mpi=pmix ./randomized_svd_multigpu_v3 \
        --m "${m}" --n "${n}" \
        "${TIME_ARGS_COMMON[@]}" \
        --compress-b-mode none \
        --compress-b-bits 0
    echo "-------------------------------------"

    echo "--- TQ 4-bit (sanity) ---"
    srun --mpi=pmix ./randomized_svd_multigpu_v3 \
        --m "${m}" --n "${n}" \
        "${TIME_ARGS_COMMON[@]}" \
        --compress-b-mode tq \
        --compress-b-bits 4 \
        --qjl-dim 0 \
        --qjl-alpha 0.0
    echo "-------------------------------------"

    echo "--- TQ+QJL 4-bit (verify post-fix) ---"
    srun --mpi=pmix ./randomized_svd_multigpu_v3 \
        --m "${m}" --n "${n}" \
        "${TIME_ARGS_COMMON[@]}" \
        --compress-b-mode tq-qjl \
        --compress-b-bits 4 \
        --qjl-dim 64 \
        --qjl-alpha 0.01
    echo "-------------------------------------"
}

run_sweep 32768 8192
run_sweep 65536 16384

echo "v3 QJL post-fix verification complete!"

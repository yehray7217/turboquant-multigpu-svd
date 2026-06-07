#!/bin/bash
# Paths are set for /work/pbr03617.
#SBATCH --job-name=rsvd_v4_oisst_smoke
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=/work/pbr03617/turboquant-multigpu-svd/randomized_svd_baseline_v4/testcases_real_world/OISST/exp_turboquant/output_logs/smoke-debug.out
#SBATCH --error=/work/pbr03617/turboquant-multigpu-svd/randomized_svd_baseline_v4/testcases_real_world/OISST/exp_turboquant/error_logs/smoke-debug.err

set -euo pipefail

module load cuda/12.8
module load ucx/1.14.1
module load openmpi/5.0.2_ucx1.14.1_cuda12.3

unset OMPI_MCA_opal_cuda_support || true
unset OMPI_MCA_btl_smcuda_use_cuda_ipc || true

PROJECT_ROOT=/work/pbr03617/turboquant-multigpu-svd/randomized_svd_baseline_v4
BIN="${PROJECT_ROOT}/.build/randomized_svd_multigpu_v4"
cd "$PROJECT_ROOT"

make
cd "${PROJECT_ROOT}/testcases_real_world/OISST/exp_turboquant"

echo "Starting OISST smoke test on matrix_debug.f32"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
nvidia-smi -L
echo "====================================="

srun --mpi=pmix "$BIN" \
    --m 8192 \
    --n 4096 \
    --k 250 \
    --oversample 6 \
    --ngpus 16 \
    --gpus-per-rank 8 \
    --seed 1234 \
    --input-file ../matrix_debug.f32 \
    --subspace-iter 1 \
    --skip-form-u \
    --compress-b-mode none \
    --compress-b-bits 0 \
    --compress-subspace-mode none \
    --compress-subspace-bits 0 \
    --summary-only \
    --repeat 50

echo "OISST smoke test complete!"

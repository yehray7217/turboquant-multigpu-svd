#!/bin/bash
# Update the absolute paths below if your clone is not under /home/rax10101010/hpc_final/.
#SBATCH --job-name=rsvd_v4_oisst_tq8
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=/home/rax10101010/hpc_final/turboquant-multigpu-svd/randomized_svd_baseline_v4/testcases_real_world/OISST/exp_turboquant/output_logs/b8-exp.out
#SBATCH --error=/home/rax10101010/hpc_final/turboquant-multigpu-svd/randomized_svd_baseline_v4/testcases_real_world/OISST/exp_turboquant/error_logs/b8-exp.err

set -euo pipefail

module load cuda/12.8
module load ucx/1.14.1
module load openmpi/5.0.2_ucx1.14.1_cuda12.3

unset OMPI_MCA_opal_cuda_support || true
unset OMPI_MCA_btl_smcuda_use_cuda_ipc || true

PROJECT_ROOT=/home/rax10101010/hpc_final/turboquant-multigpu-svd/randomized_svd_baseline_v4
BIN="${PROJECT_ROOT}/.build/randomized_svd_multigpu_v4"
cd "$PROJECT_ROOT"

make
cd "${PROJECT_ROOT}/testcases_real_world/OISST/exp_turboquant"

BASE_ARGS=(
    --m 32768
    --n 8192
    --k 250
    --oversample 6
    --ngpus 16
    --gpus-per-rank 8
    --seed 1234
    --input-file ../matrix.f32
    --subspace-iter 1
    --skip-form-u
    --compress-b-mode tq
    --compress-b-bits 8
    --compress-subspace-mode tq
    --compress-subspace-bits 8
    --summary-only
)

echo "Starting OISST experiment: TQ 8-bit"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
nvidia-smi -L
echo "====================================="

echo "Timing: TQ8"
srun --mpi=pmix "$BIN" \
    "${BASE_ARGS[@]}" \
    --repeat 50 \
    --no-check-error
echo "-------------------------------------"

echo "Final-error: TQ8"
srun --mpi=pmix "$BIN" \
    "${BASE_ARGS[@]}" \
    --repeat 50

echo "OISST TQ8 group complete!"

#!/bin/bash
#SBATCH --job-name=real_mnist_qjl_tq
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=graph_embedding_comm_%j.out
#SBATCH --error=graph_embedding_comm_%j.err

set -euo pipefail

module load cuda/12.8
module load openmpi/5.0.2_ucx1.14.1_cuda12.3

unset OMPI_MCA_opal_cuda_support || true
unset OMPI_MCA_btl_smcuda_use_cuda_ipc || true

if [ -f "${SLURM_SUBMIT_DIR}/Makefile" ]; then
    cd "${SLURM_SUBMIT_DIR}"
elif [ -f "${SLURM_SUBMIT_DIR}/graph_embedding_benchmark/Makefile" ]; then
    cd "${SLURM_SUBMIT_DIR}/graph_embedding_benchmark"
fi

make

mkdir -p results

echo "Job date: $(date)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "Running on ${SLURM_NNODES} nodes, ${SLURM_NTASKS} MPI ranks, ${SLURM_GPUS_ON_NODE:-8} GPUs/node"
nvidia-smi -L || true

echo "====================================="
echo "Part 1/2: TurboQuant communication sweep on 16 GPUs"
# Uses the original ../turboquant implementation on synthetic GNN-like
# embedding exchange payloads.
srun --mpi=pmix .build/graph_embedding_comm_benchmark \
    --halo-nodes 32768 \
    --embedding-dim 256 \
    --mode sweep \
    --repeats 10 \
    --warmup 2 | tee "results/graph_embedding_comm_${SLURM_JOB_ID}.csv"

echo "====================================="
echo "Part 2/2: real Fashion-MNIST giant-feature QJL sweep"
# Downloads real image data, expands it into an implicit 100000-D feature
# matrix, applies QJL, then compresses the QJL features with TurboQuant.
python3 prepare_mnist_data.py \
    --dataset fashion-mnist \
    --data-dir "${SLURM_SUBMIT_DIR}/data"

srun --ntasks=1 --cpus-per-task=4 .build/mnist_qjl_feature_benchmark \
    --dataset-dir "${SLURM_SUBMIT_DIR}/data/fashion-mnist" \
    --output "results/fashion_mnist_qjl_tq_${SLURM_JOB_ID}.csv" \
    --feature-dim 100000 \
    --qjl-dims 256,512,1024 \
    --tq-bits 4 \
    --train-samples 60000 \
    --test-samples 10000 \
    --feature-chunk 2048 \
    --geometry-samples 1024 \
    --geometry-pairs 4096 \
    --seed 1234

echo "====================================="
echo "All results are in:"
echo "  ${PWD}/results/graph_embedding_comm_${SLURM_JOB_ID}.csv"
echo "  ${PWD}/results/fashion_mnist_qjl_tq_${SLURM_JOB_ID}.csv"

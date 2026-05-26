#!/bin/bash
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=2                 
#SBATCH --ntasks-per-node=8        
#SBATCH --gres=gpu:8            
#SBATCH --time=00:30:00             

module purge
module load cuda/12.8
module load openmpi/4.1.6

echo "====================================="
echo "Compiling MPI + CUDA Program..."
nvcc -ccbin=mpicxx mpi_cusolver.cu -o mpi_cusolver -O3 -arch=sm_70 -Wno-deprecated-gpu-targets -lcusolver -lcudart
echo "Compilation Done."
echo "====================================="

SIZES=(1024 2048 4096 8192)

echo "Starting Distributed cuSOLVER Benchmark Sweep"
echo "Running on ${SLURM_NNODES} Nodes, $((${SLURM_NNODES} * 8)) GPUs total."
echo "Date: $(date)"
echo "====================================="

for size in "${SIZES[@]}"
do
    echo "Running SVD for Local Matrix: $size x $size"

    srun ./mpi_cusolver -M $size -N $size
    
    echo "-------------------------------------"
done

echo "Benchmark Complete!"
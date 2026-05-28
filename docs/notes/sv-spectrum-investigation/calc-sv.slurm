#!/bin/bash
#SBATCH --account=ACD115051
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=%j-calc-sv.out
#SBATCH --error=%j-calc-sv.err

set -euo pipefail

module load miniconda3/conda24.5.0_py3.9

python3 calc-sv-from-NLP-dataset.py


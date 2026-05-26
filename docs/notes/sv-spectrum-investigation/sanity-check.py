#!/bin/bash
#SBATCH --account=ACD115051
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --output=%j-calc-sv.out



pip3 --version

module spider python
module spider anaconda
module spider conda
ls /opt/conda* 2>/dev/null
ls /usr/local/anaconda* 2>/dev/null


#!/bin/bash
#SBATCH --comment=run
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --output=run.txt
#SBATCH --time=80:00:00

module load Anaconda3/2022.10
#module load cuDNN/8.0.4.30-CUDA-11.1.1

source activate SDiff

cd /users/acr22wl/TTDS/dataset

bash run.sh
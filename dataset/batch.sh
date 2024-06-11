#!/bin/bash
#SBATCH --comment=run
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --output=run.txt
#SBATCH --time=80:00:00

module load Anaconda3/2022.10

source activate SDiff

cd /users/acr22wl/TTDS/dataset

# python prepare_TORGO.py /mnt/parscratch/users/acr22wl/TTDS/dataset_2/TORGO --speechdiff_dir /users/acr22wl/TTDS/speech-diff --output_dir /mnt/parscratch/users/acr22wl/TTDS_out/output

python TTDS.py /mnt/parscratch/users/acr22wl/TTDS/dataset_2/TORGO --output_dir /mnt/parscratch/users/acr22wl/TTDS_out/output
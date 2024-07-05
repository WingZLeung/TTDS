#!/bin/bash
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --partition=dcs-gpu
#SBATCH --account=dcs-res
#SBATCH --gpus-per-node=1  # Requests 2 GPUs
#SBATCH --output=output.%{TEXT}.txt




# load modules 
module load Anaconda3/2019.07

# activate environment
source activate SD

cd /fastdata/acr22wl/TTDS_run/dataset

python TTDS.py /fastdata/acr22wl/corpora/TORGO --speechdiff_dir /fastdata/acr22wl/TTDS_run/speech-diff --output_dir /fastdata/acr22wl/TTDS_out
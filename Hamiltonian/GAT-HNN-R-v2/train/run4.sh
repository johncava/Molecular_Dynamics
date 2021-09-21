#!/bin/bash

#SBATCH -n 10                     # number of cores
#SBATCH -p gpu -q wildfire               # name of partition
#SBATCH -t 0-4:00                  # wall time (D-HH:MM)
##SBATCH -A drzuckerman             # Account hours will be pulled from (commented out with double # in front)
#SBATCH -o debug4.out             
#SBATCH -e debug4.err            
#SBATCH --mail-type=ALL             # Send a notification when the job starts, stops, or fails
#SBATCH --gres=gpu:2


module purge
module load anaconda/py3

eval "$(conda shell.bash hook)"
conda activate nichola2_pt


## just to get outside of the train scripts folder
cd ../
python train-HNN.py

conda deactivate


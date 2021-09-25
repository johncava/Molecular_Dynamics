#!/bin/bash
 
#SBATCH -N 1  # number of nodes
#SBATCH -n 8  # number of "tasks" (default: allocates 1 core per task)
#SBATCH -t 0-10:00:00   # time in d-hh:mm:ss
#SBATCH -p gpu       # partition
#SBATCH --gres=gpu:1  
#SBATCH -q wildfire       # QOS
#SBATCH -o slurm.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=nichola2@asu.edu # Mail-to address

conda activate nichola2_pt

python train-whitened.py

conda deactivate
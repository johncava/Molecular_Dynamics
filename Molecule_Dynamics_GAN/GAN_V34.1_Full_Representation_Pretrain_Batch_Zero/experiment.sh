#!/bin/bash
 
#SBATCH -N 1  # number of nodes
#SBATCH -n 8  # number of "tasks" (default: allocates 1 core per task)
#SBATCH -t 0-01:00:00   # time in d-hh:mm:ss
#SBATCH -p gpu      # partition
#SBATCH --gres=gpu:1  
#SBATCH -q wildfire       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=jcava@asu.edu # Mail-to address

source activate ~/.conda/envs/pytorch-geometric
python gan.py
conda deactivate
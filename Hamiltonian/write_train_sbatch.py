from sys import argv
### Author: Nicholas Ho
### Purpose of Script: To write sbatch scripts so multiple simulations can be conducted in parallel

## python write_sbatch chunk:1 lambd:1 ensemble_size:10 

num_submits = 10
train_folder = "train/"
email = "nichola2@asu.edu" # disabled this for now because 100 sims = 200 emails that I do not want
num_cores = 10
num_GPU = 2



top = """#!/bin/bash

#SBATCH -n %s                     # number of cores
#SBATCH -p gpu -q wildfire               # name of partition
#SBATCH -t 0-4:00                  # wall time (D-HH:MM)
##SBATCH -A drzuckerman             # Account hours will be pulled from (commented out with double # in front)
#SBATCH -o debug%s.out             
#SBATCH -e debug%s.err            
#SBATCH --mail-type=ALL             # Send a notification when the job starts, stops, or fails
#SBATCH --gres=gpu:%s
"""

body = """

module purge
module load anaconda/py3

eval "$(conda shell.bash hook)"
conda activate nichola2_pt


## just to get outside of the train scripts folder
cd ../
python train-HNN.py

conda deactivate

"""

for num in range(num_submits):
    outName = "run" + str(num) + ".sh"
    outName = train_folder + outName
    body = (body)
    header = (top % (num_cores, num, num, num_GPU))

    with open(outName, "w") as outputfile:
        outputfile.write(header)
        outputfile.write(body)

print("Finished Writing")





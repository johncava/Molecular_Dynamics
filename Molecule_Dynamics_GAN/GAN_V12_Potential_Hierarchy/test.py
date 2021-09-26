import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

###
# Important Variables
###

number_of_particles = 40
input_size = 3
hidden_size = 128
history_size = 15
lead_time = 2
M = 5
num_layers = 1
batch_size = 128

##
# Read Dataset
##
import glob
import numpy as np

files = glob.glob('./../../All_ML_Training_Data/210905_SMD_decaalanine/SMD/output/processed_orient/*.npy')

dataset = []

end_to_end_distance = dict()
for i in range(1002):
    end_to_end_distance[i] = []
    for j in range(int(40/2)):
        end_to_end_distance[i].append([])

for file_ in files:
    X_positions = np.load(file_)

    X = X_positions

    X = X[::10]

    # Create Training dataset from this sequence
    #print(X.shape[0])-> 1002
    for frame_num in range(X.shape[0]):
        dataset.append((frame_num, X[frame_num,:,:]))
        for j in range(int(40/2)):
            end_to_end_distance[frame_num][j].append(np.sqrt(np.power((X[frame_num,j,:] - X[frame_num,(40-1)-j,:]),2).sum()))

# Check the end to end distance per frame
for i in range(1002):
    for j in range(int(40/2)):
        end_to_end_distance[i][j] = np.array(end_to_end_distance[i][j]).mean().tolist()
    #end_to_end_distance[i] = np.array(end_to_end_distance[i]).mean().tolist()
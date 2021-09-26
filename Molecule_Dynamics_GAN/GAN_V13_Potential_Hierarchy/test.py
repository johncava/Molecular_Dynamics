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
# Pairwise distance matrix calculatrion (code from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/4)
##

def dist_matrix(x,y):
    x,y = torch.tensor(x), torch.tensor(y)
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, 2).sum(2) 
    return dist

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

for file_ in files:
    X_positions = np.load(file_)

    X = X_positions

    X = X[::10]

    # Create Training dataset from this sequence
    #print(X.shape[0])-> 1002
    for frame_num in range(X.shape[0]):
        dataset.append((frame_num, X[frame_num,:,:]))
        x = X[frame_num,:,:]
        end_to_end_distance[frame_num].append(dist_matrix(x,x))
        del x
    

for i in range(1002):
    end_to_end_distance[i] = torch.stack(end_to_end_distance[i])
    end_to_end_distance[i] = torch.mean(end_to_end_distance[i],0)
    print(end_to_end_distance[i].size())
    print(end_to_end_distance[i])
    break
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

##
# Arguments
##
import argparse

parser = argparse.ArgumentParser(description='GNN for Protein Dynamics Prediction.')
parser.add_argument('--data', type=str, default='/home/jcava/10_deca_alanine/99/',
                    help='directory for input data')
parser.add_argument('--num_particles', type=int, default=40, help='number of particles')
parser.add_argument('--channel_size', type=int, default=5,help='input channel size')
parser.add_argument('--hidden_size', type=int,default=3,help='hidden size')
parser.add_argument('--history_size',type=int,default=15,help='history size')
parser.add_argument('--lead_time',type=int,default=1,help='lead time')
parser.add_argument('--M',type=int,default=5,help='M number of Graph Processors')
parser.add_argument('--split',type=float,default=0.8,help='split percentage')
parser.add_argument('--lr',type=float,default=1e-2,help='learning rate')
parser.add_argument('--max_epochs',type=int,default=5,help='max epochs')

parser.add_argument('--k',type=int,default=3,help='k nearest neighbors')
parser.add_argument('--mean',type=float,default=0.0,help='mean for the input gaussian noise')
parser.add_argument('--std',type=float,default=0.01,help='std for the input gaussian noise')

parser.add_argument('--output',type=str,default='predictions',help='output name')

args = parser.parse_args()

##
# Read Dataset 
##
import glob
import numpy as np
X_positions = np.load(args.data + 'backbone.npy') #/ 17.0

#Pick the good region [5K-10K]
X = X_positions[5000:10001,:,:]

# Normalize X and Y by 7 and Z by 10
#X[:,:,:2] =  X[:,:,:2] / 7.0
#X[:,:,2:3] = X[:,:,2:3] / 10.0

# Sample down the amount of sequenced frames from 20K to 2K
#X = X[::10]
#print(X.shape)

###
# IMPORTANT VARIABLES
###

number_of_particles = args.num_particles
channel_size = args.channel_size
hidden_size = args.hidden_size
history_size = args.history_size
lead_time = args.lead_time
M = args.M
split_percentage = args.split

# Create Training dataset from this sequence
dataset = []
for i in range(X.shape[0]-(lead_time + history_size)):
    dataset.append((X[i:i+history_size,:,:], X[i+history_size+lead_time,:,:]))

# Shuffle the dataset
import random
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

split = int(len(dataset)*split_percentage)
training_dataset = dataset[:split]
testing_dataset = dataset[split:]
random.shuffle(training_dataset)

# Dataset size
print(len(training_dataset))

##
# Construct velocity histories
##
delta = 1
for data in dataset:

    x = data[0]
    y = data[1]
    #print(y.shape)
    velocity_history = []
    for index in range(history_size-1,0,-1):
        d_velocity = x[index,:,:] - x[index-1,:,:]
        #print(d_velocity.shape)
        velocity_history.append(d_velocity)
    v_i = velocity_history[-1]
    v_f = y - x[-1,:,:]
    acceleration = v_f - v_i
    velocity_history = np.concatenate(velocity_history,axis=1)
    #print(velocity_history.shape)
    new_x = np.concatenate((x[-1,:,:],velocity_history),axis=-1)
    #print(new_x.shape)
    
    print('=========')

    v_pred = v_i + delta*acceleration
    x_final = x[-1,:,:] + delta*v_pred 

    loss = np.mean((y - x_final)**2)
    print(loss)
    break
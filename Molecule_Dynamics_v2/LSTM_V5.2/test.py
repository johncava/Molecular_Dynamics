import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn 
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.data import Data 
from torch_geometric.nn import GATConv


###
# IMPORTANT VARIABLES
###

number_of_particles = 40 
input_size = 43
hidden_size = 128
history_size = 15
lead_time = 2
M = 5
num_layers = 1

##
# Read Dataset 
##
import glob
import numpy as np

files = glob.glob('./../../All_ML_Training_Data/210905_SMD_decaalanine/SMD/output/processed_orient/*.npy')

dataset = []

for file_ in files:
    X_positions = np.load(file_)

    #Pick the good region [5K-10K]
    X = X_positions

    # Sample down the amount of sequenced frames from 20K to 2K
    X = X[::10]
    #print(X.shape)

    # Create Training dataset from this sequence
    for i in range(X.shape[0]-(lead_time + history_size)):
        dataset.append((X[i:i+history_size,:,:], X[i+history_size+lead_time,:,:]))

# Shuffle the dataset
import random
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

split = int(len(dataset)*0.8)
training_dataset = dataset[:split]
testing_dataset = dataset[split:]
random.shuffle(training_dataset)

# Dataset size
print(len(training_dataset))

##
# LSTM Definition
##

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers=num_layers).cuda()
        self.h0 = torch.randn((num_layers, number_of_particles, hidden_size)).cuda()
        self.c0 = torch.randn((num_layers, number_of_particles, hidden_size)).cuda()
        self.mlp = nn.Linear(hidden_size,3).cuda()

    def reinitalize(self):
        self.h0 = torch.randn((num_layers, number_of_particles, hidden_size)).cuda()
        self.c0 = torch.randn((num_layers, number_of_particles, hidden_size)).cuda()

    def forward(self,x):
        x , (self.h0,self.c0) = self.lstm(x,(self.h0, self.c0))
        x = self.mlp(x)
        return x

##
# LSTM Initializations
##

lstm = LSTM(input_size, hidden_size)

##
# Optimization
##
import torch.optim as optim
learning_rate=1e-3
optimizer = optim.Adam(lstm.parameters(), lr=learning_rate)


##
# Pairwise distance matrix calculatrion (code from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/4)
##

def dist_matrix(x,y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, 2).sum(2) 
    return dist


##
# Create k nearest neighbors 
##
z = torch.zeros(40,40)

num_neighbors = 3
for k in range(1,num_neighbors+1):
    for i in range(40):
        if i + k + 1 <= 40:
            for j in range(i+k,i+k+1):
                z[i,j] = 1.0

z = z.float().cuda()

prediction_length = 997
predictions = []
with torch.no_grad():

    x = torch.randn(15,40,3)
    x_dist = []
    for index in range(x.size()[0]):
        x_matrix = dist_matrix(x[index,:,:],x[index,:,:])*k
        x_dist.append(torch.cat((x[index,:,:],x_matrix),1))
    
    x = torch.stack(x_dist).float().cuda()
    # Encoder
    lstm.reinitalize()
    output = lstm(x)
    output = output[-1,:,:].unsqueeze(0)
    # Decoder
    for index in range(prediction_length):
        output = output.squeeze(0)
        output_matrix = dist_matrix(output,output)*k
        output = torch.cat((output,output_matrix),1)
        output = output.unsqueeze(0)
        output = lstm(output)
        x_final = output.squeeze(0)[:,:3]
        predictions.append(x_final)

    predictions = torch.stack(predictions).squeeze(1)
    predictions = predictions.cpu().detach().numpy()
print(predictions.shape)



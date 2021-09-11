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
input_size = 3
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
        self.mlp = nn.Linear(hidden_size, 6).cuda() ### THIS IS WHAT IS CHANGED FOR NLL

    def reinitalize(self):
        self.h0 = torch.randn((num_layers, number_of_particles, hidden_size)).cuda()
        self.c0 = torch.randn((num_layers, number_of_particles, hidden_size)).cuda()

    def forward(self,x):
        x , (self.h0,self.c0) = self.lstm(x,(self.h0, self.c0))
        x = self.mlp(x)
        return x
    

##
# NLL Loss Function
##

def nll_constrained_gaussian(mu_sigma, y):
    mux = mu_sigma[:, 0]
    muy = mu_sigma[:, 1]
    muz = mu_sigma[:, 2]

    stdx = mu_sigma[:, 3]
    stdy = mu_sigma[:, 4]
    stdz = mu_sigma[:, 5]

    truex = y[:, 0]
    truey = y[:, 1]
    truez = y[:, 2]

    squarex = (mux - truex) ** 2
    msx = squarex / (stdx**2) + torch.log(stdx**2)

    squarey = (muy - truey) **2
    msy = squarey / (stdy**2) + torch.log(stdy**2)

    squarez = (muz - truez) **2
    msz = squarez / (stdz**2) + torch.log(stdz**2)
    
    minimize_this = torch.mean(msx) + torch.mean(msy) + torch.mean(msz)
    
    return minimize_this
    


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
# Run Training
##

max_epochs = 5

epoch_loss = []

print("Starting Training...")

import time
start = time.time()
for epoch in range(max_epochs):

    training_loss = []
    for data in training_dataset:
        #print(data[0].shape,data[1].shape)
        x = torch.tensor(data[0]).float().cuda()
        #x_final = x[-1,:,:3]
        y = torch.tensor(data[1]).float().cuda()
        # LSTM
        lstm.reinitalize()
        output = lstm(x)
        # Loss computation
        optimizer.zero_grad()
        y = y[:,:3]
#         loss = F.mse_loss(output[-1,:,:], y)
        mu_sigma = output[-1,:,:]
        loss = nll_constrained_gaussian(mu_sigma, y)
        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        
    epoch_loss.append(np.mean(training_loss))
    print('Epoch ' + str(epoch) + ' Loss: ' + str(epoch_loss[-1]))
end = time.time()
print('Done in ' + str(end-start) + 's')


PATH = "trained_LSTMNLL_Model.pt"
pt.save(model.state_dict(), PATH)
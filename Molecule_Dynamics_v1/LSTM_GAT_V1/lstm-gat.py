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
parser.add_argument('--lead_time',type=int,default=2,help='lead time')
parser.add_argument('--num_layers',type=int,default=1,help='number of layers of LSTM')
parser.add_argument('--k',type=int,default=3,help='k nearest neighbors')
parser.add_argument('--split',type=float,default=0.8,help='split percentage')
parser.add_argument('--lr',type=float,default=1e-2,help='learning rate')
parser.add_argument('--max_epochs',type=int,default=5,help='max epochs')

#parser.add_argument('--mean',type=float,default=0.0,help='mean for the input gaussian noise')
#parser.add_argument('--std',type=float,default=0.01,help='std for the input gaussian noise')

parser.add_argument('--output',type=str,default='predictions',help='output name')

args = parser.parse_args()

##
# Read Dataset 
##
import glob
import numpy as np
X_positions = np.load(args.data + 'backbone.npy') #/ 17.0
X_angles = np.load(args.data + 'allPhiPsi.npy')

# Reshape X_angles to get every 10th frame (200000, number of particles, 2) => (20000, number of particles, 2)
X_angles = X_angles[::10] #/ 180.0

# Concatenate the PhiPsi angles to the XYZ cartesian coordinates
X = np.concatenate((X_positions, X_angles), axis = -1)
print(X.shape)

#Pick the good region [5K-10K]
X = X[5000:10001,:,:]

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
input_size = args.channel_size
hidden_size = args.hidden_size
history_size = args.history_size
lead_time = args.lead_time
num_layers = args.num_layers

# Create Training dataset from this sequence
dataset = []
for i in range(X.shape[0]-(lead_time + history_size)):
    dataset.append((X[i:i+history_size,:,:], X[i+history_size+lead_time,:,:]))

# Shuffle the dataset
import random
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

split = int(len(dataset)*args.split)
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

    def forward(self,x):
        self.h0 = torch.randn((num_layers, number_of_particles, hidden_size)).cuda()
        self.c0 = torch.randn((num_layers, number_of_particles, hidden_size)).cuda()
        x , _ = self.lstm(x,(self.h0, self.c0))
        return x

##
# GAT Defintion
##

class GATDecoder(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(GATDecoder, self).__init__()
        self.gat1 = GATConv(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        return self.gat1(x, edge_index)

##
# LSTM and GAT Initializations
##

lstm = LSTM(input_size, hidden_size)
gat_decoder = GATDecoder(hidden_size, hidden_size).cuda()

##
# Optimization
##
import torch.optim as optim
learning_rate=args.lr
optimizer = optim.Adam(list(lstm.parameters()) + list(gat_decoder.parameters()), lr=learning_rate)

##
# Run Training
##

max_epochs = args.max_epochs

epoch_loss = []

import time
start = time.time()
for epoch in range(max_epochs):

    training_loss = []
    for data in training_dataset:
        #print(data[0].shape,data[1].shape)
        x = torch.tensor(data[0]).float().cuda()
        x_final = x[-1,:,:3]
        y = torch.tensor(data[1]).float().cuda()
        # GAT Encoder
        output = lstm(x)
        output = output[-1,:,:]
        output = Data(pos=output).cuda()
        transform = T.KNNGraph(k=args.k)
        output = transform(output)
        output = gat_decoder(output.pos, output.edge_index)
        x_final = x_final + output
        #print(output[-1,:,:].max())
        x_final = x_final
        # Loss computation
        optimizer.zero_grad()
        y = y[:,:3]
        loss = F.mse_loss(x_final, y)
        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    epoch_loss.append(np.mean(training_loss))
    print('Epoch ' + str(epoch) + ' Loss: ' + str(epoch_loss[-1]))
end = time.time()
print('Done in ' + str(end-start) + 's')

##
# Run Testing
##
predictions = []
for data in testing_dataset:
    # LSTM Encoder
    x = torch.tensor(data[0]).float().cuda()
    x_final = x[-1,:,:3]
    y = torch.tensor(data[1]).float().cuda()
    # GAT Encoder
    output = lstm(x)
    output = output[-1,:,:]
    output = Data(pos=output).cuda()
    transform = T.KNNGraph(k=args.k)
    output = transform(output)
    output = gat_decoder(output.pos, output.edge_index)
    x_final = x_final + output
    x_final = x_final
    predictions.append(x_final)

predictions = torch.stack(predictions).squeeze(1)
predictions = predictions.cpu().detach().numpy()
print(predictions.shape)

#predictions[:,:,:2] =  predictions[:,:,:2] * 7.0
#predictions[:,:,2:3] = predictions[:,:,2:3] * 10.0
# Save predictions
np.save(args.output + ".npy", predictions)

# Save predictions into VMD format
frame_num = predictions.shape[0]

nAtoms = "40"
outName = args.output + ".xyz"
with open(outName, "w") as outputfile:
    for frame_idx in range(frame_num):
        
        frame = predictions[frame_idx]
        outputfile.write(str(nAtoms) + "\n")
        outputfile.write(" generated by JK\n")

        atomType = "CA"
        for i in range(40):
            line = str(frame[i][0]) + " " + str(frame[i][1]) + " " + str(frame[i][2]) + " "
            line += "\n"
            outputfile.write("  " + atomType + "\t" + line)

print("=> Finished Generation <=")
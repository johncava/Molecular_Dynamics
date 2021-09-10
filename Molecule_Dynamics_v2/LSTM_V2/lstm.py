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
# Run Training
##

max_epochs = 5

epoch_loss = []

import time
start = time.time()
for epoch in range(max_epochs):

    training_loss = []
    for data in training_dataset:
        #print(data[0].shape,data[1].shape)
        x = torch.tensor(data[0]).float().cuda()
        #x_final = x[-1,:,:3]
        y = torch.tensor(data[1]).float().cuda()
        # GAT Encoder
        lstm.reinitalize()
        output = lstm(x)
        # Construct the end to end distance (e.g atom 1 - 40, 2 - 39, 3 - 38, ..., etc) for X
        end_to_end_distance_x = []
        output = output[-1,:,:3]
        for i in range(int(number_of_particles/2)):
            a = output[i,:3]
            b = output[i,:3]
            dist = torch.cdist(a.view(1,3),b.view(1,3),p=2).view(1)
            end_to_end_distance_x.append(dist)
        end_to_end_distance_x = torch.stack(end_to_end_distance_x)
        # Construct end to end distance for Y
        y = y[:,:3]
        end_to_end_distance_y = []
        for i in range(int(number_of_particles/2)):
            a = y[i,:3]
            b = y[i,:3]
            dist = torch.cdist(a.view(1,3),b.view(1,3),p=2).view(1)
            end_to_end_distance_y.append(dist)
        end_to_end_distance_y = torch.stack(end_to_end_distance_y)
        # Loss computation
        optimizer.zero_grad()
        y = y[:,:3]
        loss = F.mse_loss(output, y) + F.mse_loss(end_to_end_distance_x, end_to_end_distance_y)
        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    epoch_loss.append(np.mean(training_loss))
    print('Epoch ' + str(epoch) + ' Loss: ' + str(epoch_loss[-1]))
end = time.time()
print('Done in ' + str(end-start) + 's')

##
# Run Testing Non-AutoRegressively
##

X_positions = np.load('/home/jcava/10_deca_alanine/99/backbone.npy') #/ 17.0

# Reshape X_angles to get every 10th frame (200000, number of particles, 2) => (20000, number of particles, 2)

#Pick the good region [5K-10K]
X = X_positions[5000:10001,:,:]

testing_dataset = []
for i in range(X.shape[0]-(lead_time + history_size)):
    testing_dataset.append((X[i:i+history_size,:,:], X[i+history_size+lead_time,:,:]))

predictions = []
for data in testing_dataset:
    # LSTM Encoder
    x = torch.tensor(data[0]).float().cuda()
    #x_final = x[-1,:,:3]
    y = torch.tensor(data[1]).float().cuda()
    # GAT Encoder
    lstm.reinitalize()
    output = lstm(x)
    #x_final = x_final + output[-1,:,:]
    #x_final = x_final
    predictions.append(output[-1,:,:])

predictions = torch.stack(predictions).squeeze(1)
predictions = predictions.cpu().detach().numpy()
print(predictions.shape)

#predictions[:,:,:2] =  predictions[:,:,:2] * 7.0
#predictions[:,:,2:3] = predictions[:,:,2:3] * 10.0
# Save predictions
np.save("predictions.npy", predictions)

# Save predictions into VMD format
frame_num = predictions.shape[0]

nAtoms = "40"
outName = "predictions.xyz"
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

##
# Run Testing Auto-Regressively
##

X_positions = np.load('/home/jcava/10_deca_alanine/99/backbone.npy') #/ 17.0

# Reshape X_angles to get every 10th frame (200000, number of particles, 2) => (20000, number of particles, 2)

#Pick the good region [5K-10K]
X = X_positions[5000:10001,:,:]

testing_dataset = []
for i in range(X.shape[0]-(lead_time + history_size)):
    testing_dataset.append((X[i:i+history_size,:,:], X[i+history_size+lead_time,:,:]))

prediction_length = 997
predictions = []
with torch.no_grad():

    x = torch.tensor(testing_dataset[0][0]).float().cuda()
    # Encoder
    lstm.reinitalize()
    output = lstm(x)
    output = output[-1,:,:].unsqueeze(0)
    for index in range(prediction_length):
        output = output + torch.randn((1,number_of_particles,3)).cuda()
        output, (h0,c0) = decoder(output, (h0,c0))
        #output = output.squeeze(0)
        x_final = output[:,:3].squeeze(0)
        predictions.append(x_final)

    predictions = torch.stack(predictions).squeeze(1)
    predictions = predictions.cpu().detach().numpy()
print(predictions.shape)

# Save predictions
np.save("predictions-auto.npy", predictions)

# Save predictions into VMD format
frame_num = predictions.shape[0]

nAtoms = "40"
outName = "predictions-auto.xyz"
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
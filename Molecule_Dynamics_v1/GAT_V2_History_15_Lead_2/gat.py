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
# Read Dataset 
##
import glob
import numpy as np
X_positions = np.load('/home/jcava/10_deca_alanine/99/backbone.npy') #/ 17.0
X_angles = np.load('/home/jcava/10_deca_alanine/99/allPhiPsi.npy')

# Reshape X_angles to get every 10th frame (200000, number of particles, 2) => (20000, number of particles, 2)
X_angles = X_angles[::10] / 180.0

# Concatenate the PhiPsi angles to the XYZ cartesian coordinates
X = np.concatenate((X_positions, X_angles), axis = -1)
print(X.shape)

#Pick the good region [5K-10K]
X = X[5000:10001,:,:]

# Normalize X and Y by 7 and Z by 10
X[:,:,:2] =  X[:,:,:2] / 7.0
X[:,:,2:3] = X[:,:,2:3] / 10.0

# Sample down the amount of sequenced frames from 20K to 2K
#X = X[::10]
#print(X.shape)

###
# IMPORTANT VARIABLES
###

number_of_particles = 40 
channel_size = 5
hidden_size = 3
history_size = 15
lead_time = 2
M = 5

# Create Training dataset from this sequence
dataset = []
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
# GAT Definition
##

class GATEncoder(torch.nn.Module):

    def __init__(self,in_channels, out_channels):
        super(GATEncoder,self).__init__()
        self.gat1 = GATConv(in_channels, out_channels)

    def forward(self,x, edge_index):
        return self.gat1(x, edge_index).relu()

class GATDecoder(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(GATDecoder, self).__init__()
        self.gat1 = GATConv(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        return self.gat1(x, edge_index).relu()

class GATProcessor(torch.nn.Module):

    def __init__(self, channels, M):
        super(GATProcessor, self).__init__()
        self.processor = [GATEncoder(channels, channels).cuda() for _ in range(M)]

    def forward(self, x_i):
        for p in self.processor:
            x_i = Data(pos=x_i)
            transform = T.KNNGraph(k=3)
            x_i = transform(x_i)
            x_i = x_i.pos + p(x_i.pos, x_i.edge_index).relu()
        return x_i

##
# GAT Initializations
##
gat_encoder = GATEncoder(channel_size, hidden_size).cuda()
gat_decoder = GATDecoder(hidden_size, hidden_size).cuda()
gat_processor = GATProcessor(hidden_size, 5)


##
# Optimization
##
import torch.optim as optim
gat_processor_params = []
for p in gat_processor.processor:
    gat_processor_params += list(p.parameters())
learning_rate=1e-2
optimizer = optim.Adam(list(gat_encoder.parameters()) + list(gat_decoder.parameters()) +
                        gat_processor_params, lr=learning_rate)

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
        x_final = x[-1,:,:3]
        y = torch.tensor(data[1]).float().cuda()
        # GAT Encoder
        x_encoded = torch.zeros((number_of_particles, hidden_size)).cuda()
        for i in range(x.size()[0]):
            x_i = x[i,:,:]
            x_i = Data(pos=x_i).cuda()
            transform = T.KNNGraph(k=3)
            x_i = transform(x_i)
            x_encoded += gat_encoder(x_i.pos, x_i.edge_index)

        # GAT Processor
        processed = gat_processor(x_encoded)
        
        # GAT Decoder 
        processed = Data(pos=processed)
        transform = T.KNNGraph(k=3)
        processed = transform(processed)

        decoded = gat_decoder(processed.pos, processed.edge_index)
        #print(decoded.size())
        x_final = x_final + decoded
        x_final = x_final.tanh()
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
    x = torch.tensor(data[0]).float().cuda()
    x_final = x[-1,:,:3]
    # GAT Encoder
    x_encoded = torch.zeros((number_of_particles, hidden_size)).cuda()
    for i in range(x.size()[0]):
        x_i = x[i,:,:]
        x_i = Data(pos=x_i).cuda()
        transform = T.KNNGraph(k=3)
        x_i = transform(x_i)
        x_encoded += gat_encoder(x_i.pos, x_i.edge_index)

    # GAT Processor
    processed = gat_processor(x_encoded)
    
    # GAT Decoder 
    processed = Data(pos=processed)
    transform = T.KNNGraph(k=3)
    processed = transform(processed)

    decoded = gat_decoder(processed.pos, processed.edge_index)
    x_final = x_final + decoded
    x_final = x_final.tanh()
    predictions.append(x_final)

predictions = torch.stack(predictions).squeeze(1)
predictions = predictions.cpu().detach().numpy()
print(predictions.shape)

predictions[:,:,:2] =  predictions[:,:,:2] * 7.0
predictions[:,:,2:3] = predictions[:,:,2:3] * 10.0
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

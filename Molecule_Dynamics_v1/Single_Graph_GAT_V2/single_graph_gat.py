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
#parser.add_argument('--channel_size', type=int, default=42,help='input channel size')
parser.add_argument('--hidden_size', type=int,default=64,help='hidden size')
parser.add_argument('--output_size',type=int,default=3,help='output size')
parser.add_argument('--history_size',type=int,default=10,help='history size')
parser.add_argument('--lead_time',type=int,default=1,help='lead time')
parser.add_argument('--M',type=int,default=5,help='M number of Graph Processors')
parser.add_argument('--split',type=float,default=0.8,help='split percentage')
parser.add_argument('--lr',type=float,default=1e-2,help='learning rate')
parser.add_argument('--max_epochs',type=int,default=10,help='max epochs')

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
X = X[::10]
#print(X.shape)

###
# IMPORTANT VARIABLES
###

number_of_particles = args.num_particles
channel_size = (args.history_size-1)*3 + 3
hidden_size = args.hidden_size
history_size = args.history_size
output_size = args.output_size
lead_time = args.lead_time
M = args.M
split_percentage = args.split

# Create Training dataset from this sequence
dataset = []
for i in range(X.shape[0]-(lead_time + history_size)):
    dataset.append((X[i:i+history_size,:,:], X[i+history_size+lead_time,:,:]))

##
# Construct velocity histories
##

new_dataset = []
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
    v_i = velocity_history[0]
    v_f = y - x[-1,:,:]
    acceleration = v_f - v_i
    velocity_history = np.concatenate(velocity_history,axis=1)
    #print(velocity_history.shape)
    #new_x = np.concatenate((x[-1,:,:],velocity_history),axis=-1)
    #print(new_x.shape)
    
    '''
    ##
    # SANITY CHECK
    ##

    print('=========')

    v_pred = v_i + delta*acceleration
    x_final = x[-1,:,:] + delta*v_pred 

    loss = np.mean((y - x_final)**2)
    print(loss)
    '''
    new_dataset.append((x[-1,:,:],velocity_history, acceleration,y))

# Shuffle the dataset
import random
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

split = int(len(new_dataset)*split_percentage)
copy_dataset = new_dataset[::]
training_dataset = new_dataset[:split]
testing_dataset = new_dataset[split:]
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
    
    def forward(self, x, edge_index):
        return self.gat1(x=x, edge_index=edge_index).sigmoid()


class GATDecoder(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(GATDecoder, self).__init__()
        self.gat1 = GATConv(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        return self.gat1(x=x, edge_index=edge_index)

class GATProcessor(torch.nn.Module):

    def __init__(self, channels, M):
        super(GATProcessor, self).__init__()
        self.processor = [GATEncoder(channels, channels).cuda() for _ in range(M)]

    def forward(self, x, edge_index):
        for p in self.processor:
            x_i = x + p(x, edge_index).sigmoid()
        return x_i

##
# GAT Initializations
##
gat_encoder = GATEncoder(channel_size, hidden_size).cuda()
gat_decoder = GATDecoder(hidden_size, output_size).cuda()
gat_processor = GATProcessor(hidden_size, hidden_size)

##
# Optimization
##
import torch.optim as optim
gat_processor_params = []
for p in gat_processor.processor:
    gat_processor_params += list(p.parameters())
learning_rate= args.lr
optimizer = optim.Adam(list(gat_encoder.parameters()) + list(gat_decoder.parameters()) +
                        gat_processor_params, lr=learning_rate)


max_epochs = args.max_epochs

epoch_loss = []

import time
start = time.time()
for epoch in range(max_epochs):

    training_loss = []
    for data in training_dataset:

        ##
        # Data Input Processing
        ##

        x,velocity_history,acceleration,y = data
        #print(x.shape,velocity_history.shape,acceleration.shape,y.shape)

        X_pos = x[::]
        velocity_history = np.concatenate((X_pos, velocity_history),axis=-1)
        x = torch.tensor(x).float().cuda()
        velocity_history = torch.tensor(velocity_history).float().cuda()
        x = Data(x=velocity_history,pos=x)
        transform0 = T.KNNGraph(k=40)
        x = transform0(x)
        transform = T.Distance(norm=True)
        x = transform(x)

        #x.x = x.x.pow(2)

        acceleration = torch.tensor(acceleration).float().cuda()
        #print(x)

        ##
        # GAT Encoder
        ##
        x_encoded = gat_encoder(x.x, x.edge_index)

        ##
        # GAT Processor
        ##
        x_processed = gat_processor(x_encoded, x.edge_index)

        ##
        # GAT Decoder
        ##
        decoded = gat_decoder(x_processed, x.edge_index)

        pred_acceleration = decoded 

        # Loss computation
        optimizer.zero_grad()
        loss = F.mse_loss(pred_acceleration,acceleration)
        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    epoch_loss.append(np.mean(training_loss))

    ##
    # Save the models per each epoch
    ##

    torch.save(gat_encoder.state_dict(),'./gat_encoder-' + str(epoch) + '.pt')
    torch.save(gat_processor.state_dict(),'./gat_processor-' + str(epoch) + '.pt')
    torch.save(gat_decoder.state_dict(),'./gat_decoder-' + str(epoch) + '.pt')
    
    print('Epoch ' + str(epoch) + ' Loss: ' + str(epoch_loss[-1]))
end = time.time()
print('Done in ' + str(end-start) + 's')


##
# Run Testing
##
predictions = []
for data in copy_dataset:
    
    ##
    # Data Input Processing
    ##

    x,velocity_history,acceleration,y = data
    #print(x.shape,velocity_history.shape,acceleration.shape,y.shape)

    X_pos = x[::]
    velocity_history = np.concatenate((X_pos, velocity_history),axis=-1)
    x = torch.tensor(x).float().cuda()
    velocity_history = torch.tensor(velocity_history).float().cuda()
    v_i = velocity_history[:,3:6]
    x = Data(x=velocity_history,pos=x)
    transform0 = T.KNNGraph(k=40)
    x = transform0(x)
    transform = T.Distance(norm=True)
    x = transform(x)

    #x.x = x.x.pow(2)

    ##
    # GAT Encoder
    ##
    x_encoded = gat_encoder(x.x, x.edge_index)

    ##
    # GAT Processor
    ##
    x_processed = gat_processor(x_encoded, x.edge_index)

    ##
    # GAT Decoder
    ##
    decoded = gat_decoder(x_processed, x.edge_index)

    pred_acceleration = decoded 
    v_pred = v_i + delta*pred_acceleration
    x_final = x.pos + delta*v_pred
    predictions.append(x_final)

predictions = torch.stack(predictions).squeeze(1)
predictions = predictions.cpu().detach().numpy()
print(predictions.shape)

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

##
# Save models
##

#torch.save(gat_encoder.state_dict(),'./gat_encoder.pt')
#torch.save(gat_processor.state_dict(),'./gat_processor.pt')
#torch.save(gat_decoder.state_dict(),'./gat_decoder.pt')

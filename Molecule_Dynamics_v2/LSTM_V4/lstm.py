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
input_size = 5
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

files = glob.glob('./../../All_ML_Training_Data/210905_SMD_decaalanine/SMD/output/processed_orient_phi_psi/*.npy')

dataset = []

for file_ in files:
    X_positions = np.load(file_)

    #Pick the good region [5K-10K]
    X = X_positions

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
learning_rate=1e-4
optimizer = optim.Adam(lstm.parameters(), lr=learning_rate)


def dihedral2(p):
    b = p[:-1] - p[1:]
    b[0] *= -1
    v = [ v - (v.dot(b[1])/b[1].dot(b[1])) * b[1] for v in [b[0], b[2]] ] 
    v = torch.stack(v)
    # Normalize vectors
    v /= torch.sqrt(torch.einsum('...i,...i', v, v)).view(-1,1)
    b1 = b[1] / torch.linalg.norm(b[1])
    x = torch.dot(v[0], v[1])
    m = torch.cross(v[0], b1)
    y = torch.dot(m, v[1])
    return torch.rad2deg(torch.atan2( y, x ))


def getPhiVals(frame):

    phi_indices = np.array([
    [3, 5, 6, 7], #4
    [3, 5, 6, 7], #8
    [7, 9, 10, 11],  #12
    [11, 13, 14, 15], # 16
    [15, 17, 18, 19], # 20
    [19, 21, 22, 23], # 24
    [23, 25, 26, 27], # 28
    [27, 29, 30, 31], # 32
    [31, 33, 34, 35], # 36
    [35, 39, 40, 37] # 40 ugh 
    ])

    phi_vals = []
    for i in range(len(phi_indices)):
        sel = phi_indices[i] - 1
        chosen_frames = torch.stack((frame[sel[0]], frame[sel[1]], frame[sel[2]], frame[sel[3]]))
        phi = dihedral2(chosen_frames)
        phi_vals.append(phi)

    return torch.tensor([phi_vals]), phi_indices


def getPsiVals(frame):

    ## N CA C N
    psi_indices = np.array([
        [1, 2, 3, 5],
        [5, 6, 7, 9],
        [9, 10, 11, 13],
        [13, 14, 15, 17],
        [17, 18, 19, 21],
        [21, 22, 23, 25],
        [25, 26, 27, 29],
        [29, 30, 31, 33],
        [33, 34, 35, 39],
        [33, 34, 35, 39]
    ])

    psi_vals = []
    for i in range(len(psi_indices)):
        sel = psi_indices[i] - 1
        chosen_frames = torch.stack((frame[sel[0]], frame[sel[1]], frame[sel[2]], frame[sel[3]]))
        psi = dihedral2(chosen_frames)
        psi_vals.append(psi)

    return torch.tensor([psi_vals]), psi_indices

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
        x = torch.tensor(data[0]).float().cuda()
        y = torch.tensor(data[1]).float().cuda()
        # GAT Encoder
        lstm.reinitalize()
        output = lstm(x)
        output = output[-1,:,:]
        # Loss computation
        optimizer.zero_grad()
        y_pos = y[:,:3]
        position_loss = F.mse_loss(output, y_pos)
        output_phi,_ = getPhiVals(output)
        output_psi,_ = getPsiVals(output)
        y_phi,_ = getPhiVals(y_pos)
        y_psi,_ = getPsiVals(y_pos)
        loss = position_loss + F.mse_loss(output_phi,y_phi) + F.mse_loss(output_psi,y_phi)
        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    epoch_loss.append(np.mean(training_loss))
    print('Epoch ' + str(epoch) + ' Loss: ' + str(epoch_loss[-1]))
end = time.time()
print('Done in ' + str(end-start) + 's')

'''
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

'''

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

    x = torch.tensor(testing_dataset[0][0]).float()
    x_phis, x_psis = torch.zeros(number_of_particles,1), torch.zeros(number_of_particles,1)
    new_x = []
    for s in range(x.size()[0]):
        x_phi, phi_index = getPhiVals(x[s,:,:])
        x_psi, psi_index = getPsiVals(x[s,:,:])
        for i in range(phi_index.shape[0]):
            indices = [p-1 for p in phi_index[i].tolist()]
            x_phis[indices] = x_phi[0][i]
        for j in range(psi_index.shape[0]):
            indices = [p-1 for p in psi_index[j].tolist()]
            x_psis[indices] = x_psi[0][j]
        new_x.append(torch.cat((x[s,:,:],x_phis,x_psis),1))
    new_x = torch.stack(new_x).numpy().tolist()
    x = torch.tensor(new_x).float().cuda()
    # Encoder
    lstm.reinitalize()
    output = lstm(x)
    output = output[-1,:,:].unsqueeze(0)
    x_phis, x_psis = torch.zeros(number_of_particles,1).cuda(), torch.zeros(number_of_particles,1).cuda()
    new_x = []
    for s in range(output.size()[0]):
        x_phi, phi_index = getPhiVals(output[s,:,:])
        x_psi, psi_index = getPsiVals(output[s,:,:])
        for i in range(phi_index.shape[0]):
            indices = [p-1 for p in phi_index[i].tolist()]
            x_phis[indices] = x_phi[0][i].cuda()
        for j in range(psi_index.shape[0]):
            indices = [p-1 for p in psi_index[j].tolist()]
            x_psis[indices] = x_psi[0][j].cuda()
        new_x.append(torch.cat((output[s,:,:],x_phis,x_psis),1))
    output = torch.stack(new_x).float().cuda()
    # Decoder
    for index in range(prediction_length):
        output = lstm(output)
        x_final = output.squeeze(0)[:,:3]
        predictions.append(x_final)
        x_phis, x_psis = torch.zeros(number_of_particles,1).cuda(), torch.zeros(number_of_particles,1).cuda()
        new_x = []
        for s in range(output.size()[0]):
            x_phi, phi_index = getPhiVals(output[s,:,:])
            x_psi, psi_index = getPsiVals(output[s,:,:])
            for i in range(phi_index.shape[0]):
                indices = [p-1 for p in phi_index[i].tolist()]
                x_phis[indices] = x_phi[0][i].cuda()
            for j in range(psi_index.shape[0]):
                indices = [p-1 for p in psi_index[j].tolist()]
                x_psis[indices] = x_psi[0][j].cuda()
            new_x.append(torch.cat((output[s,:,:],x_phis,x_psis),1))
        output = torch.stack(new_x).float().cuda()

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
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

    dataset.append(X)

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

for k,data in enumerate(dataset):
    x = torch.tensor(data).float()
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
    new_x = torch.stack(new_x).detach().numpy()
    print(str(k), new_x.shape)
    np.save('./../../All_ML_Training_Data/210905_SMD_decaalanine/SMD/output/processed_orient_phi_psi/' + str(k) + '.npy',new_x)
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
X_angles = X_angles[::10] #/ 180.0

# Concatenate the PhiPsi angles to the XYZ cartesian coordinates
X = np.concatenate((X_positions, X_angles), axis = -1)
#X = X_positions
print(X.shape)

#Pick the good region [5K-10K]
X = X[5000:10001,:,:]
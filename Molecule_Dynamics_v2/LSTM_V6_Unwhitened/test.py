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
# Make the k-nearest neighbor dist matrix
##

a = torch.randn(40,3)

def dist_matrix(x,y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, 2).sum(2) 
    return dist

b = dist_matrix(a,a)

z = torch.zeros(40,40)

num_neighbors = 3
for k in range(1,num_neighbors+1):
    for i in range(40):
        if i + k + 1 <= 40:
            for j in range(i+k,i+k+1):
                z[i,j] = 1.0
zz = b*z
print(zz)
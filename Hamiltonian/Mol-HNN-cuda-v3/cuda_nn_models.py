# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch
import numpy as np
from cuda_utils import choose_nonlinearity

# class MLP(torch.nn.Module):
#   '''Just a salt-of-the-earth MLP'''
#   def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
#     super(MLP, self).__init__()
#     self.linear1 = torch.nn.Linear(input_dim, hidden_dim).cuda()
#     self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim).cuda()
#     self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=None).cuda()

#     for l in [self.linear1, self.linear2, self.linear3]:
#       torch.nn.init.orthogonal_(l.weight).cuda() # use a principled initialization

#     self.nonlinearity = choose_nonlinearity(nonlinearity)

#   def forward(self, x, separate_fields=False):
#     h = self.nonlinearity( self.linear1(x) )
#     h = self.nonlinearity( self.linear2(h) )
#     return self.linear3(h)


class MLP(torch.nn.Module):
  '''Just a salt-of-the-earth MLP'''
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
    super(MLP, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim).cuda()
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim).cuda()
    self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim).cuda()
    self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim).cuda()
    self.linear5 = torch.nn.Linear(hidden_dim, hidden_dim).cuda()
    self.linear6 = torch.nn.Linear(hidden_dim, output_dim, bias=None).cuda()

    for l in [self.linear1, self.linear2, self.linear3]:
      torch.nn.init.orthogonal_(l.weight).cuda() # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x, separate_fields=False):
    h = self.nonlinearity( self.linear1(x) )
    h = self.nonlinearity( self.linear2(h) )
    h = self.nonlinearity( self.linear3(h) )
    h = self.nonlinearity( self.linear4(h) )
    h = self.nonlinearity( self.linear5(h) )
    return self.linear6(h)


class MLPAutoencoder(torch.nn.Module):
  '''A salt-of-the-earth MLP Autoencoder + some edgy res connections'''
  def __init__(self, input_dim, hidden_dim, latent_dim, nonlinearity='tanh'):
    super(MLPAutoencoder, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear4 = torch.nn.Linear(hidden_dim, latent_dim)

    self.linear5 = torch.nn.Linear(latent_dim, hidden_dim)
    self.linear6 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear7 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear8 = torch.nn.Linear(hidden_dim, input_dim)

    for l in [self.linear1, self.linear2, self.linear3, self.linear4, \
              self.linear5, self.linear6, self.linear7, self.linear8]:
      torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def encode(self, x):
    h = self.nonlinearity( self.linear1(x) )
    h = h + self.nonlinearity( self.linear2(h) )
    h = h + self.nonlinearity( self.linear3(h) )
    return self.linear4(h)

  def decode(self, z):
    h = self.nonlinearity( self.linear5(z) )
    h = h + self.nonlinearity( self.linear6(h) )
    h = h + self.nonlinearity( self.linear7(h) )
    return self.linear8(h)

  def forward(self, x):
    z = self.encode(x)
    x_hat = self.decode(z)
    return x_hat


class Cfconv(torch.nn.Module):

  def __init__(self,position_dim,hidden_dim,num_particles):
    super(Cfconv,self).__init__()
    self.dense1 = torch.nn.Conv1d(num_particles, hidden_dim,1)
    self.dense2 = torch.nn.Conv1d(hidden_dim, hidden_dim,1)

  def forward(self, x, r):
    positions = torch.cdist(r,r).exp()
    positions = self.dense1(positions)
    positions = torch.log(0.5*torch.exp(positions) + 0.5)
    positions = self.dense2(positions)
    positions = torch.log(0.5*torch.exp(positions) + 0.5)
    x = x * positions
    return x, positions

class InteractionBlock(torch.nn.Module):

  def __init__(self,num_particles):
    super(InteractionBlock,self).__init__()
    self.cfconv = Cfconv(64,64,num_particles)
    self.atomwise1 = torch.nn.Conv1d(64,64,1)
    self.atomwise2 = torch.nn.Conv1d(64,64,1)

  def forward(self, x, r):
    x = self.atomwise1(x)
    x,r = self.cfconv(x,r)
    x = self.atomwise2(x)
    x = torch.log(0.5*torch.exp(x) + 0.5)
    x = x * r
    return x

class SchNet(torch.nn.Module):

  def __init__(self,num_particles):
    super(SchNet,self).__init__()
    self.interaction1 = InteractionBlock(num_particles)
    self.interaction2 = InteractionBlock(num_particles)
    self.interaction3 = InteractionBlock(num_particles)
    self.embedding = torch.nn.Conv1d(3,64,1)
    self.atomwise1 = torch.nn.Conv1d(64,32,1)
    self.atomwise2 = torch.nn.Conv1d(32,2,1)

  def forward(self, x):
    x = x[:,120:].view(x.size()[0],3,40)
    r = x[:,:120].view(x.size()[0],40,3)
    x = self.embedding(x)
    #print(x.size())
    x = self.interaction1(x,r)
    #print(x.size())
    x = self.interaction2(x,r)
    #print(x.size())
    x = self.interaction3(x,r)
    #print(x.size())
    x = self.atomwise1(x)
    x = torch.log(0.5*torch.exp(x) + 0.5)
    x = self.atomwise2(x)
    x = x.view(x.size()[0],40,2).sum(dim=1)
    return x

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

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
    self.processor = torch.nn.ModuleList([GATEncoder(channels, channels).cuda() for _ in range(M)])

  def forward(self, x, edge_index):
    for p in self.processor:
      x = x + p(x, edge_index)
    return x

class GATModel(torch.nn.Module):

  def __init__(self,channel_size, hidden_size, output_size):
    super(GATModel, self).__init__()
    self.gat_encoder = GATEncoder(channel_size, hidden_size)
    self.gat_decoder = GATDecoder(hidden_size, output_size)
    self.gat_processor = GATProcessor(hidden_size, 3)
    self.transform0 = T.KNNGraph(k=40)
    self.transform = T.Distance(norm=True)
  
  def forward(self,x):
    x = x[:,120:].view(x.size()[0],40,3)
    r = x[:,:120].view(x.size()[0],40,3)
    
    outputs = []
    for i in range(x.size()[0]):
      xi = Data(x=x[i,:,:], pos=r[i,:,:])
      xi = self.transform0(xi)
      xi = self.transform(xi)

      ###
      # GAT Encoder
      ###
      x_encoded = self.gat_encoder(xi.x, xi.edge_index)

      ###
      # GAT Processor
      ###
      x_processed = self.gat_processor(x_encoded, xi.edge_index)

      ###
      # GAT Decoder
      ###
      x_decoded = self.gat_decoder(x_processed, xi.edge_index)

      # x output
      output = x_decoded.sum(dim=0)
      outputs.append(output)
    outputs = torch.stack(outputs)
    return outputs



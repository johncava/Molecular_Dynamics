from torchdyn.core import NeuralODE
from torchdyn.models import HNN
from torchdyn.nn import DataControl, DepthCat, Augmenter
from torchdyn.datasets import *
from torchdyn.utils import *

# sentinel boolean to run once 
dry_run = False

import torch
import torch.nn as nn

class HNN(nn.Module):

    def __init__(self, Hamiltonian:nn.Module, dim=1):
        super().__init__()
        self.H = Hamiltonian
        self.n = dim

    def forward(self, x):
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            gradH = torch.autograd.grad(self.H(x).sum(), x, allow_unused=False, create_graph=True)[0]
            return torch.cat([gradH[:,self.n:], -gradH[:,:self.n]], 1).to(x)

##
# Data input from the Hamiltonian Tutorial
##
import torch.utils.data as data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

t = torch.linspace(0,1, 100).reshape(-1,1)
X = torch.cat([
    torch.sin(2*np.pi*t),
    torch.cos(2*np.pi*t)
],1).to(device)

y = torch.cat([
    torch.cos(2*np.pi*t),
    -torch.sin(2*np.pi*t)
],1).to(device)

train = data.TensorDataset(X, y)
trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)

###
# Pytorch Lightning Learner
###
import pytorch_lightning as pl

class Learner(pl.LightningModule):
    def __init__(self, model:nn.Module):
        super().__init__()
        self.model = model
        self.c = 0
    
    def forward(self, x):
        return self.model.defunc(0,x)
    
    def loss(self, y, y_hat):
        return ((y-y_hat)**2).sum()
    
    def training_step(self, batch, batch_idx):
        x, y = batch      
        y_hat = self.model.defunc(0,x)   
        loss = self.loss(y_hat, y)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}   
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_dataloader(self):
        return trainloader


###
# Initialize HamFunc
###

HamFunc = HNN(nn.Sequential(
            nn.Linear(2,64),
            nn.Tanh(),
            nn.Linear(64,1))).to(device)


model = NeuralODE(HamFunc, sensitivity='adjoint',solver='dopri5').to(device)

###
# Run Training
###
learn = Learner(model)
trainer = pl.Trainer(min_epochs=500, max_epochs=1000)
trainer.fit(learn)

print('Done')
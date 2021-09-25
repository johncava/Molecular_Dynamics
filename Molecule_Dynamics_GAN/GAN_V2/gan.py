import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

###
# Important Variables
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

    X = X_positions

    X = X[::10]

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

generator = LSTM(input_size, hidden_size)

##
# Optimization
##
import torch.optim as optim
learning_rate=1e-3
optimizer = optim.Adam(generator.parameters(), lr=learning_rate)

##
# Run (MLE) Pre-training of LSTM Generator Model
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
        generator.reinitalize()
        output = generator(x)
        output = output[-1,:,:]
        # Loss computation
        optimizer.zero_grad()
        y = y[:,:3]
        loss = F.mse_loss(output, y)
        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        break
    epoch_loss.append(np.mean(training_loss))
    break
    print('Epoch ' + str(epoch) + ' Loss: ' + str(epoch_loss[-1]))
end = time.time()
print('Done in ' + str(end-start) + 's')

###
# Discriminator Definition
###

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.mlp1 = nn.Conv1d(40,5,1)
        self.mlp2 = nn.Conv1d(5,1,1)
        self.mlp3 = nn.Linear(3,1)

    def forward(self,x):
        x = torch.sigmoid(self.mlp1(x))
        x = torch.sigmoid(self.mlp2(x))
        x = x.view(x.size()[0],3)
        x = torch.sigmoid(self.mlp3(x))
        return x

##
# Discriminator defintions
##

discriminator = Discriminator().cuda()

###
# SchNet for (v) Potential Loss Function
###
class Cfconv(torch.nn.Module):

  def __init__(self,position_dim,hidden_dim,num_particles):
    super(Cfconv,self).__init__()
    self.dense1 = torch.nn.Conv1d(num_particles, hidden_dim,1)
    self.dense2 = torch.nn.Conv1d(hidden_dim, hidden_dim,1)

  def forward(self, x, r):
    positions = torch.cdist(r,r)
    positions = torch.exp(-positions)
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
    self.atomwise2 = torch.nn.Conv1d(32,1,1)

  def forward(self, x_input):
    x = x_input.view(x_input.size()[0],3,40)
    r = x_input.detach().clone().view(x_input.size()[0],40,3)
    x = self.embedding(x)
    x = self.interaction1(x,r)
    x = self.interaction2(x,r)
    x = self.interaction3(x,r)
    x = self.atomwise1(x)
    x = torch.log(0.5*torch.exp(x) + 0.5)
    x = self.atomwise2(x)
    x = x.view(x.size()[0],40,1).sum(dim=1)
    return x

##
# Discriminator defintions
##

invariant_function = SchNet(number_of_particles).cuda()

###
# Begin GAN Training
###
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
d_optimizer = optim.Adam(discriminator.parameters(),lr=learning_rate)

# Initalize BCELoss function
criterion = nn.BCELoss()

for epoch in range(max_epochs):

    training_loss = []
    for data in training_dataset:
        ###
        # (1) Update D Network: maximize log(D(x)) + log (1 - D(G(z)))
        ###
        discriminator.zero_grad()
        label = torch.ones((1,)).float().cuda()
        x = torch.tensor(data[1]).unsqueeze(0).float().cuda()
        pred = discriminator(x).squeeze(0)
        d_real = criterion(pred, label)
        d_real.backward() 

        # Train with fake examples
        x = torch.tensor(data[0]).float().cuda()
        # Generator
        generator.reinitalize()
        output = generator(x)
        output = output[-1,:,:].unsqueeze(0)
        # D(G(z)))
        pred = discriminator(output).squeeze(0)
        label = torch.zeros((1,)).float().cuda()
        d_fake = criterion(pred, label)
        d_fake.backward()
        # Update discriminator weights after loss backward from BOTH d_real AND d_fake examples
        d_optimizer.step()

        ###
        # (2) Update G Network: maximize log(D(G(z)))
        ###
        g_optimizer.zero_grad()
        label = torch.ones((1,)).float().cuda()
        x = torch.tensor(data[0]).float().cuda()
        # Generator
        generator.reinitalize()
        output = generator(x)
        output = output[-1,:,:].unsqueeze(0)
        # D(G(z)))
        pred = discriminator(output).squeeze(0)
        label =  torch.zeros((1,)).float().cuda()
        g_fake = criterion(pred, label)
        g_fake.backward()
        # Update generator weights
        g_optimizer.step()

        ###
        # (3) Update G Network: minimize log(I(G(z)))
        ###
        g_optimizer.zero_grad()
        x = torch.tensor(data[0]).float().cuda()
        # Generator
        generator.reinitalize()
        output = generator(x)
        output = output[-1,:,:].unsqueeze(0).view(1,120)
        # D(G(z)))
        invariant_loss = invariant_function(output)
        #print(invariant_loss)
        invariant_loss.backward()
        # Update generator weights
        g_optimizer.step()
        break
    break

print('Done Done')

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
batch_size = 128

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
    #print(X.shape[0])-> 1002
    for frame_num in range(X.shape[0]):
        dataset.append((frame_num, X[frame_num,:,:]))

new_dataset = []
for batch in range(int(len(dataset)/batch_size)):

    batched = []
    for item in dataset[batch*batch_size:batch*batch_size + batch_size]:

        fnum, data = item
        new_data = np.concatenate([[fnum], data.reshape(120)],-1).reshape(121)
        batched.append(new_data)

    batched = np.stack(batched)
    new_dataset.append(batched)

dataset = new_dataset
#print(dataset[0].shape)
#print(dataset[0])

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
# Generator Definition
##

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.mlp1 = nn.Linear(32,50)
        self.mlp2 = nn.Linear(50,100)
        self.mlp3 = nn.Linear(100,120)

    def forward(self,batch_size, max_steps):
        t = random.choices(range(max_steps),k=batch_size)
        t = torch.tensor(t).view(batch_size,1)
        t = t/max_steps 
        t = t.float().cuda()
        z = torch.normal(0,1,size=(batch_size,31)).cuda()
        z = torch.cat((t,z),1)
        z = torch.sigmoid(self.mlp1(z))
        z = torch.sigmoid(self.mlp2(z))
        z = self.mlp3(z)
        return t, z 

    def generation_step(self, t, max_steps):
        t = torch.tensor(t).view(1,1)
        t = t/max_steps 
        t = t.float().cuda()
        z = torch.normal(0,1,size=(batch_size,31)).cuda()
        z = torch.cat((t,z),1)
        z = torch.sigmoid(self.mlp1(z))
        z = torch.sigmoid(self.mlp2(z))
        z = self.mlp3(z)
        return z
##
# LSTM Initializations
##

generator = Generator().cuda()

###
# Discriminator Definition
###

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.mlp1 = nn.Linear(121, 50)
        self.mlp2 = nn.Linear(50, 32)
        self.mlp3 = nn.Linear(32,1)

    def forward(self,x):
        x = torch.sigmoid(self.mlp1(x))
        x = torch.sigmoid(self.mlp2(x))
        x = torch.sigmoid(self.mlp3(x))
        return x

##
# Discriminator defintions
##

discriminator = Discriminator().cuda()

###
# Begin GAN Training
###
import torch.optim as optim
learning_rate = 1e-3

g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
d_optimizer = optim.Adam(discriminator.parameters(),lr=learning_rate)

# Initalize BCELoss function
criterion = nn.BCELoss()

max_epochs = 10
Ng = 5
Nd = 5

for epoch in range(max_epochs):

    training_loss = []
    for data in training_dataset:
        
        ###
        # (1) Update D Network: maximize log(D(x)) + log (1 - D(G(z)))
        ###

        for _ in range(Nd):
          discriminator.zero_grad()
          label = torch.ones((batch_size,1)).float().cuda()
          x = torch.tensor(data).float().cuda()
          pred = discriminator(x).squeeze(0)
          d_real = criterion(pred, label)
          d_real.backward() 

          # Train with fake examples
          # Generator
          t,output = generator(batch_size,1002)
          output = torch.cat([t,output],1)
          # D(G(z)))
          pred = discriminator(output).squeeze(0)
          label = torch.zeros((batch_size,1)).float().cuda()
          d_fake = criterion(pred, label)
          d_fake.backward()
          # Update discriminator weights after loss backward from BOTH d_real AND d_fake examples
          d_optimizer.step()

        ###
        # (2) Update G Network: maximize log(D(G(z)))
        ###
        for _ in range(Ng):
          g_optimizer.zero_grad()
          # Generator
          t, output = generator(batch_size,1002)
          output = torch.cat([t,output],1)
          # D(G(z)))
          pred = discriminator(output).squeeze(0)
          label =  torch.ones((batch_size,1)).float().cuda()
          g_fake = criterion(pred, label)
          g_fake.backward()
          # Update generator weights
          g_optimizer.step()

print('Done')


##
# Generation
##
generator.eval()
# Go through the reaction coordinate of the trajectory
max_generation_steps = 1000
predictions = []
for t in range(max_generation_steps):
    gen_frame = generator.generation_step(t, max_generation_steps)
    predictions.append(gen_frame.view(40,3))

predictions = torch.stack(predictions)
predictions = predictions.cpu().detach().numpy()
# Save predictions into VMD format
predictions = frames
frame_num = predictions.shape[0]

nAtoms = "40"
outName = "GAN.xyz"
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
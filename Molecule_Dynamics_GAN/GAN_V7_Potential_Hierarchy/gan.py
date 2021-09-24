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

buckets = 100

a = [(bucket*buckets,bucket*buckets+buckets) for bucket in range(10)]

def find(d):
    for index, item in enumerate(a):
        if item[0] < d and item[1] > d:
                return index
    return -1

for file_ in files:
    X_positions = np.load(file_)

    X = X_positions

    X = X[::10]

    # Create Training dataset from this sequence
    #print(X.shape[0])-> 1002
    for frame_num in range(X.shape[0]):
        f = find(frame_num)
        if f != -1:
            dataset.append((f,X[frame_num,:,:]))

# Shuffle the dataset
import random
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

random.shuffle(dataset)

new_dataset = []
for batch in range(int(len(dataset)/batch_size)):

    batched = []
    for item in dataset[batch*batch_size:batch*batch_size + batch_size]:

        bnum, data = item
        h = torch.tensor([bnum])
        h = F.one_hot(h,num_classes=10)
        new_data = np.concatenate([h, data.reshape(1,120)],-1).reshape(130)
        batched.append(new_data)

    batched = np.stack(batched)
    new_dataset.append(batched)

training_dataset = new_dataset


# Dataset size
print(len(training_dataset))

##
# Generator Definition
##

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.mlp1 = nn.Linear(40,50)
        self.mlp2 = nn.Linear(50,100)
        self.mlp3 = nn.Linear(100,120)

    def forward(self,batch_size, max_steps):
        t = torch.tensor(random.choices(range(max_steps),k=batch_size))
        t = F.one_hot(t,num_classes=max_steps)
        t = t.float().cuda()
        z = torch.normal(0,1,size=(batch_size,30)).cuda()
        z = torch.cat((t,z),1)
        z = torch.sigmoid(self.mlp1(z))
        z = torch.sigmoid(self.mlp2(z))
        z = self.mlp3(z)
        return t, z 

    def generation_step(self, t, max_steps):
        t = torch.tensor([t])
        t = F.one_hot(t,num_classes=max_steps)
        t = t/max_steps 
        t = t.float().cuda()
        z = torch.normal(0,1,size=(1,31)).cuda()
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
        self.mlp1 = nn.Linear(130, 50)
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
# (v) Potential Loss Function
###
from moleculekit.molecule import Molecule
import os
import numpy as np
from torchmd.forcefields.forcefield import ForceField
from torchmd.parameters import Parameters
import torch
from torchmd.integrator import maxwell_boltzmann
from torchmd.systems import System
from torchmd.forces import Forces

# Define Class
class Energy:
    UNITS = "kcal/mol"

    def __init__(self, data_dir, psf_file, parameter_file, device="cuda:0", precision=torch.float, etype='all'):
        self.etype = etype
        # Make Molecule object
        mol = Molecule(os.path.join(data_dir, psf_file))  # Reading the system topology
        self.num_atoms = mol.numAtoms
        # Create Force Field object
        ff = ForceField.create(mol, os.path.join(data_dir, parameter_file))
        parameters = Parameters(ff, mol, precision=precision)
        # My Nvidia driver was too old thus I disabled the gpu
        if device == None:
            self.parameters = Parameters(ff, mol, precision=precision)
        else:
            self.parameters = Parameters(ff, mol, precision=precision, device=device)
        # Convert Moleculekit Molecule object to torchmd system object
        self.system = System(self.num_atoms, nreplicas=1, precision=precision, device=device)

        

    def __str__(self):
        return f"Energy type is {self.etype} in units of {self.UNITS}"

    def calc_energy(self, coords):
        '''Calc energies with torchmd given a set of coordinates'''
        # Reshape array if needed
        '''
        if not coords.shape == (self.num_atoms, 3, 1):
            coords = np.reshape(coords, (self.num_atoms, 3, 1))
        '''
        # Set positions for system object
        self.system.set_positions(coords)
        # Evaluate current energy and forces. Forces are modified in-place
        forces = Forces(self.parameters, cutoff=9, rfa=True, switch_dist=7.5)
        Epot = forces.compute(self.system.pos, self.system.box, self.system.forces, returnDetails=True)
        if self.etype == 'all':
            energies = Epot
        else:
            energies = Epot[self.etype]
        return energies

##
# Configurations for Energy Calculation
##
data_dir = "./../../V_Calculations/Test-3_energy_module/data/"
psf_file = "backbone-no-improp.psf"  # This is a special psf file with improper connectivity deleted
parameter_file = "param_bb-3.0.yaml" # bond, angles, dihedrals, electrostatics, lj; no 1-4, impropers or external
# Make energy calculation object



###
# Begin GAN Training
###
import torch.optim as optim
learning_rate = 1e-3

g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
d_optimizer = optim.Adam(discriminator.parameters(),lr=learning_rate)

# Initalize BCELoss function
criterion = nn.BCELoss()

max_epochs = 3
Ng = 5
Nd = 5
Ni = 5

potential_keys = [['bonds'], ['bonds','angles'], ['bonds','angles','dihedrals']]
for epoch in range(max_epochs):

    training_loss = []
    for data in training_dataset:
        ###
        # (2) Update G Network: maximize log(D(G(z)))
        ###
        g_optimizer.zero_grad()
        # Generator
        t,output = generator(batch_size,10)
        output = torch.cat([t,output],1)
        # D(G(z)))
        pred = discriminator(output).squeeze(0)
        label =  torch.ones((batch_size,1)).float().cuda()
        g_fake = criterion(pred, label)
        g_fake.backward()
        # Update generator weights
        g_optimizer.step()

        del t
        del output 

        ###
        # (1) Update D Network: maximize log(D(x)) + log (1 - D(G(z)))
        ###
        discriminator.zero_grad()
        label = torch.ones((batch_size,1)).float().cuda()
        x = torch.tensor(data).float().cuda()
        pred = discriminator(x).squeeze(0)
        d_real = criterion(pred, label)
        d_real.backward() 


        # Train with fake examples
        # Generator
        t,output = generator(batch_size,10)
        output = torch.cat([t,output],1)
        # D(G(z)))
        pred = discriminator(output).squeeze(0)
        label = torch.zeros((batch_size,1)).float().cuda()
        d_fake = criterion(pred, label)
        d_fake.backward()
        # Update discriminator weights after loss backward from BOTH d_real AND d_fake examples
        d_optimizer.step()

        del t
        del output

        ###
        # (3) Update G Network: minimize log(I(G(z)))
        ###
        for i in range(2):
            g_optimizer.zero_grad()
            # Generator
            t,output = generator(1,10)
            output = output.view(1,120).view(40,3,1)
            # D(G(z)))
            #output = output.detach().cpu()
            sys_decal = Energy(data_dir, psf_file, parameter_file)  
            potential = sys_decal.calc_energy(output)
            total_pot = torch.zeros(1,1).cuda()
            for key in potential_keys[epoch]:
                total_pot += potential[0][key]
            # Update generator weights
            total_pot.backward()
            clipping_value = 1 # arbitrary value of your choosing
            torch.nn.utils.clip_grad_norm_(generator.parameters(), clipping_value)
            g_optimizer.step()

            del sys_decal
            del t, output
            del potential
            del total_pot

print('Done Done')

##
# Generation
##
generator.eval()
# Go through the reaction coordinate of the trajectory
max_generation_steps = 10
predictions = []
for t in range(max_generation_steps):
    gen_frame = generator.generation_step(t, max_generation_steps)
    predictions.append(gen_frame.view(40,3))

predictions = torch.stack(predictions)
predictions = predictions.cpu().detach().numpy()
# Save predictions into VMD format
frame_num = predictions.shape[0]

nAtoms = "40"
outName = "GAN_2.xyz"
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
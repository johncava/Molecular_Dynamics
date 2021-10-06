##
# GAN_V10.1: End to end distance is used as a discriminator feature instead of direct loss
##
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

end_to_end_distance = dict()
for i in range(1002):
    end_to_end_distance[i] = []

for file_ in files:
    X_positions = np.load(file_)

    X = X_positions

    X = X[::10]

    # Create Training dataset from this sequence
    #print(X.shape[0])-> 1002
    for frame_num in range(X.shape[0]):
        dataset.append((frame_num, X[frame_num,:,:]))
        end_to_end_distance[frame_num].append(np.sqrt(np.power((X[frame_num,0,:] - X[frame_num,-1,:]),2).sum()))


# Check the end to end distance per frame
for i in range(1002):
    end_to_end_distance[i] = np.array(end_to_end_distance[i]).mean().tolist()
    #print(end_to_end_distance[i])


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

#print(end_to_end_distance)

# Shuffle the dataset
import random
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

random.shuffle(dataset)
training_dataset = dataset


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
        picked_t = t
        t = torch.tensor(t).view(batch_size,1)
        t = t/max_steps 
        t = t.float().cuda()
        z = torch.normal(0,1,size=(batch_size,31)).cuda()
        z = torch.cat((t,z),1)
        z = torch.sigmoid(self.mlp1(z))
        z = torch.sigmoid(self.mlp2(z))
        z = self.mlp3(z)
        return t, z, picked_t

    def generation_step(self, t, max_steps):
        t = torch.tensor(t).view(1,1)
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
        self.mlp1 = nn.Linear(122, 50)
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
# Import modules
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

    def __init__(self, data_dir, psf_file, parameter_file,
                 colvar=None, device="cuda:0", precision=torch.float, etype='all'):
        self.etype = etype
        # Make Molecule object
        mol = Molecule(os.path.join(data_dir, psf_file))  # Reading the system topology
        self.num_atoms = mol.numAtoms
        # Create Force Field object
        ff = ForceField.create(mol, os.path.join(data_dir, parameter_file))
        parameters = Parameters(ff, mol, precision=precision)
        # My Nvidia driver was too old thus I disabled the gpu
        self.dtype = torch.float
        if device == None:
            self.device = torch.device("cpu")
            self.parameters = Parameters(ff, mol, precision=precision)
        else:
            self.device = torch.device("cuda:0")
            self.parameters = Parameters(ff, mol, precision=precision, device=device)
        # Convert Moleculekit Molecule object to torchmd system object
        self.system = System(self.num_atoms, nreplicas=1, precision=precision, device=device)
        if not colvar == None:
            self.colvar_name = colvar['name']
            self.colvar_fk = torch.tensor(colvar['fk'], device=self.device, dtype=self.dtype)
            self.colvar_cent_0 = torch.tensor(colvar['cent_0'], device=self.device, dtype=self.dtype)
            self.colvar_cent_1 = torch.tensor(colvar['cent_1'], device=self.device, dtype=self.dtype)
            self.colvar_T = torch.tensor(colvar['T'], device=self.device, dtype=self.dtype)
            self.colvar_group1 = colvar['group1']
            self.colvar_group2 = colvar['group2']
        

    def __str__(self):
        return f"Energy type is {self.etype} in units of {self.UNITS}"

    
    def calc_energy(self, coords, time=None):
        '''Calc energies with torchmd given a set of coordinates'''
        # Set positions for system object
        self.system.set_positions(coords)
        # Evaluate current energy and forces. Forces are modified in-place
        forces = Forces(self.parameters, cutoff=9, rfa=True, switch_dist=7.5)
        Epot = forces.compute(self.system.pos, self.system.box, self.system.forces, returnDetails=True)
        if not self.colvar_name == None:
            if time == None:
                print("No time provided.  Exiting calculation.")
                exit()
            time = torch.tensor(time).float().cuda()
            cur_center = ((self.colvar_cent_1-self.colvar_cent_0)/self.colvar_T)*time + self.colvar_cent_0
            # print(self.system.pos[0][0])
            grp1_com = self.system.pos[0][self.colvar_group1[0]]
            grp2_com = self.system.pos[0][self.colvar_group2[0]]
            dist = torch.pow(torch.sum(torch.pow(torch.sub(grp2_com, grp1_com),2)),0.5)
            engy = torch.mul(torch.mul(torch.pow(torch.sub(cur_center, dist),2), self.colvar_fk),0.5)
            # force = torch.mul(torch.sub(cur_center, dist), self.colvar_fk)
            Epot = Epot[0]
            Epot[self.colvar_name] = engy
            Epot = [Epot]
            
        if self.etype == 'all':
            energies = Epot
        else:
            energies = Epot[0][self.etype]
            energies = [energies]
        return energies


colvar = {
    "name": "E2End Harm",
    "fk": 1.0,
    "cent_0": 12.0,
    "cent_1": 34.0,
    "T": 500000/50,
    "group1": [0],
    "group2": [39]
}

##
# Configurations for Energy Calculation
##
data_dir = "./../../V_Calculations/Test-5_bias_n_improper/data/"
psf_file = "backbone.psf"  # This is a special psf file with improper connectivity deleted
parameter_file = "param_bb-4.0.yaml" # bond, angles, dihedrals, electrostatics, lj; no 1-4, impropers or external
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

max_epochs = 5
Ng = 5
Nd = 10
Ni = 5

potential_keys = [['bonds','angles','dihedrals','impropers'] for _ in range(max_epochs)]
potential_factors = [(0.5,0.2,0.1, 0.1)] + [(0.2,0.2,0.2,0.2) for _ in range(max_epochs)]

generator_loss = []
discriminator_loss = []
potential_loss = []

for epoch in range(max_epochs):

    training_loss = []
    for data in training_dataset:
        ###
        # (2) Update G Network: maximize log(D(G(z)))
        ###
        g_optimizer.zero_grad()
        # Generator
        t,output,_ = generator(batch_size,1002)
        output_cp = output.view(batch_size,40,3)
        batched_dist = []
        for b in range(output_cp.size()[0]):
            dis = (output_cp[b,0,:] - output_cp[b,-1,:]).pow(2).sum().sqrt()
            batched_dist.append(dis)
        batched_dist = torch.stack(batched_dist)
        batched_dist = batched_dist.view(batch_size,1)
        output = torch.cat([t,output,batched_dist],1)
        # D(G(z)))
        pred = discriminator(output).squeeze(0)
        label =  torch.ones((batch_size,1)).float().cuda()
        g_fake = criterion(pred, label)
        generator_loss.append(g_fake.item())
        g_fake.backward()
        # Update generator weights
        g_optimizer.step()

        del t
        del output 

        for _ in range(Nd):
            ###
            # (1) Update D Network: maximize log(D(x)) + log (1 - D(G(z)))
            ###
            discriminator.zero_grad()
            label = torch.ones((batch_size,1)).float().cuda()
            x = torch.tensor(data).float().cuda()
            output_cp = x[:,1:].view(batch_size,40,3)
            batched_dist = []
            for b in range(output_cp.size()[0]):
                dis = (output_cp[b,0,:] - output_cp[b,-1,:]).pow(2).sum().sqrt()
                batched_dist.append(dis)
            batched_dist = torch.stack(batched_dist)
            batched_dist = batched_dist.view(batch_size,1)
            x = torch.cat([x, batched_dist],1)
            pred = discriminator(x).squeeze(0)
            d_real = criterion(pred, label)
            d_real.backward() 

            # Train with fake examples
            # Generator
            t,output,_ = generator(batch_size,1002)
            output_cp = output.view(batch_size,40,3)
            batched_dist = []
            for b in range(output_cp.size()[0]):
                dis = (output_cp[b,0,:] - output_cp[b,-1,:]).pow(2).sum().sqrt()
                batched_dist.append(dis)
            batched_dist = torch.stack(batched_dist)
            batched_dist = batched_dist.view(batch_size,1)
            output = torch.cat([t,output,batched_dist],1)
            # D(G(z)))
            pred = discriminator(output).squeeze(0)
            label = torch.zeros((batch_size,1)).float().cuda()
            d_fake = criterion(pred, label)
            discriminator_loss.append(d_fake.item() + d_real.item())
            d_fake.backward()
            # Update discriminator weights after loss backward from BOTH d_real AND d_fake examples
            d_optimizer.step()

            del x
            del output_cp 
            del t
            del output


        ###
        # (3) Update G Network: minimize log(I(G(z)))
        ###
        p_factors = potential_factors[epoch]
        bonds_factor = torch.tensor(p_factors[0]).float().cuda()
        angle_factor = torch.tensor(p_factors[1]).float().cuda()
        dihedral_factor = torch.tensor(p_factors[2]).float().cuda()
        improper_factor = torch.tensor(p_factors[3]).float().cuda()
        for i in range(2):
            g_optimizer.zero_grad()
            # Generator
            _,output,t = generator(1,1002)
            output = output.view(1,120).view(40,3,1)
            # D(G(z)))
            #output = output.detach().cpu()
            sys_decal = Energy(data_dir, psf_file, parameter_file, colvar=colvar)  
            potential = sys_decal.calc_energy(output,t)
            total_pot = torch.zeros(1,1).cuda()
            for key in potential_keys[epoch]:
                if key == 'bonds':
                    total_pot += bonds_factor*potential[0][key]
                elif key == 'angles':
                    total_pot += angle_factor*potential[0][key]
                elif key == 'dihedrals':
                    total_pot += dihedral_factor*potential[0][key]
                elif key == 'impropers':
                    total_pot += improper_factor*potential[0][key]
            # Update generator weights
            total_pot += potential[0]['E2End Harm']
            potential_loss.append(total_pot.item())
            total_pot.backward()
            clipping_value = 1 # arbitrary value of your choosing
            torch.nn.utils.clip_grad_norm_(generator.parameters(), clipping_value)
            g_optimizer.step()

            del sys_decal
            del t, output
            del potential
            del total_pot
            
        del bonds_factor
        del angle_factor
        del dihedral_factor
        del improper_factor

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

##
# Plot Loss
##
plt.plot(range(len(discriminator_loss)), discriminator_loss)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.savefig('discriminator_loss.png')

plt.plot(range(len(generator_loss)), generator_loss)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.savefig('generator_loss.png')

plt.plot(range(len(potential_loss)), potential_loss)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.savefig('potential_loss.png')

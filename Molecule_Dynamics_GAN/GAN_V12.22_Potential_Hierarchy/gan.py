##
# GAN_V12: Added target end to end distance of atom 1 and atom 30, atom 2 and atom 39, etc..., averaged for each frame from the 200 trajectory dataset
# mse loss on dis_target is used in the same time as the update on the generator for potential energy
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
for i in range(100):
    end_to_end_distance[i] = []
    for j in range(int(40/2)):
        end_to_end_distance[i].append([])

for file_ in files:
    X_positions = np.load(file_)

    X = X_positions[:1000]

    X = X[::10]

    # Create Training dataset from this sequence
    #print(X.shape[0])-> 100
    for frame_num in range(X.shape[0]):
        dataset.append((frame_num, X[frame_num,:,:]))
        for j in range(int(40/2)):
            end_to_end_distance[frame_num][j].append(np.sqrt(np.power((X[frame_num,j,:] - X[frame_num,(40-1)-j,:]),2).sum()))

# Check the end to end distance per frame
for i in range(100):
    for j in range(int(40/2)):
        end_to_end_distance[i][j] = np.array(end_to_end_distance[i][j]).mean().tolist()
    #end_to_end_distance[i] = np.array(end_to_end_distance[i]).mean().tolist()

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
        self.mlp1 = nn.Linear(141, 50)
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

max_epochs = 30
Ng = 5
Nd = 5
Ni = 5

potential_keys = [['bonds','angles','dihedrals'], ['bonds','angles','dihedrals','impropers'], 
                  ['bonds','angles','dihedrals', 'impropers', 'lj'], ['bonds','angles','dihedrals', 'impropers', 'lj', 'electrostatics']]
potential_factors = [(0.6,0.2,0.1),(0.2, 0.2, 0.2, 0.3),(0.1,0.1,0.1,0.1,0.5), (0.1,0.1,0.1,0.1,0.1,0.4)]

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
        t,output,_ = generator(batch_size,100)
        batch_dist = []
        output_cp = output.view(batch_size,40,3)
        for b in range(output_cp.size()[0]):
            pred_dist = []
            for i in range(int(40/2)):
                pred_dist.append((output_cp[b,i,:] - output_cp[b,(40-1)-i,:]).pow(2).sum().sqrt())
            pred_dist = torch.stack(pred_dist).view(1,20)
            batch_dist.append(pred_dist)
        batch_dist = torch.stack(batch_dist).view(batch_size,20)
        output = torch.cat([t,output,batch_dist],1)
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


        ###
        # (1) Update D Network: maximize log(D(x)) + log (1 - D(G(z)))
        ###
        discriminator.zero_grad()
        label = torch.ones((batch_size,1)).float().cuda()
        x = torch.tensor(data).float().cuda()
        batch_dist = []
        output_cp = x[:,1:].view(batch_size,40,3)
        for b in range(output_cp.size()[0]):
            pred_dist = []
            for i in range(int(40/2)):
                pred_dist.append((output_cp[b,i,:] - output_cp[b,(40-1)-i,:]).pow(2).sum().sqrt())
            pred_dist = torch.stack(pred_dist).view(1,20)
            batch_dist.append(pred_dist)
        batch_dist = torch.stack(batch_dist).view(batch_size,20)
        x = torch.cat([x,batch_dist],1)
        pred = discriminator(x).squeeze(0)
        d_real = criterion(pred, label)
        d_real.backward() 

        # Train with fake examples
        # Generator
        t,output,_ = generator(batch_size,100)
        output_cp = output.view(batch_size,40,3)
        batch_dist = []
        for b in range(output_cp.size()[0]):
            pred_dist = []
            for i in range(int(40/2)):
                pred_dist.append((output_cp[b,i,:] - output_cp[b,(40-1)-i,:]).pow(2).sum().sqrt())
            pred_dist = torch.stack(pred_dist).view(1,20)
            batch_dist.append(pred_dist)
        batch_dist = torch.stack(batch_dist).view(batch_size,20)
        output = torch.cat([t,output, batch_dist],1)
        # D(G(z)))
        pred = discriminator(output).squeeze(0)
        label = torch.zeros((batch_size,1)).float().cuda()
        d_fake = criterion(pred, label)
        discriminator_loss.append(d_fake.item() + d_real.item())
        d_fake.backward()
        # Update discriminator weights after loss backward from BOTH d_real AND d_fake examples
        d_optimizer.step()

        del t
        del output


        ###
        # (3) Update G Network: minimize log(I(G(z)))
        ###
        p_factors = potential_factors[epoch]
        bonds_factor = torch.tensor(p_factors[0]).float().cuda()
        angle_factor = torch.tensor(p_factors[1]).float().cuda()
        dihedral_factor = torch.tensor(p_factors[2]).float().cuda()
        improper_factor = None
        lj_factor =  None
        electro_factor =  None

        for _ in range(2):
            g_optimizer.zero_grad()
            # Generator
            _,output,t = generator(1,100)
            output = output.view(1,120).view(40,3,1)
            # D(G(z)))
            #output = output.detach().cpu()
            sys_decal = Energy(data_dir, psf_file, parameter_file)  
            potential = sys_decal.calc_energy(output)
            total_pot = torch.zeros(1,1).cuda()
            for key in potential_keys[epoch]:
                if key == 'bonds':
                    total_pot += bonds_factor*potential[0][key]
                elif key == 'angles':
                    total_pot += angle_factor*potential[0][key]
                elif key == 'dihedrals':
                    total_pot += dihedral_factor*potential[0][key]
                elif key == 'impropers':
                    improper_factor = torch.tensor(p_factors[3]).float().cuda()
                    total_pot += improper_factor*potential[0][key]
                elif key == 'lj':
                    lj_factor =  torch.tensor(p_factors[4]).float().cuda()
                    total_pot += lj_factor*potential[0][key]
                elif key == 'electrostatics':
                    electro_factor =  torch.tensor(p_factors[5]).float().cuda()
                    total_pot += electro_factor*potential[0][key]
            # Update generator weights
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
        del lj_factor
        del electro_factor
        
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
outName = "GAN_4.xyz"
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
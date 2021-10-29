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

number_of_particles = 104
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

files = glob.glob('./../../All_ML_Training_Data/210905_SMD_decaalanine/SMD/output/rand_orein_npy_traj/*.npy')

dataset = []

end_to_end_distance = dict()
for i in range(1002):
    end_to_end_distance[i] = []

pretrain_dataset = []
for file_ in files:
    X_positions = np.load(file_)

    X = X_positions

    X = X[::10]

    # Create Training dataset from this sequence
    #print(X.shape[0])-> 1002
    for frame_num in range(X.shape[0]):
        dataset.append((frame_num, X[frame_num,:,:]))
        #end_to_end_distance[frame_num].append(np.sqrt(np.power((X[frame_num,0,:] - X[frame_num,-1,:]),2).sum()))
        if frame_num < 10:
            pretrain_dataset.append((frame_num, X[frame_num,:,:]))
'''
# Check the end to end distance per frame
for i in range(1002):
    end_to_end_distance[i] = np.array(end_to_end_distance[i]).mean().tolist()
    #print(end_to_end_distance[i])
'''

new_dataset = []
for batch in range(int(len(dataset)/batch_size)):

    batched = []
    for item in dataset[batch*batch_size:batch*batch_size + batch_size]:

        fnum, data = item
        new_data = np.concatenate([[fnum], data.reshape(312)],-1).reshape(313)
        batched.append(new_data)

    batched = np.stack(batched)
    new_dataset.append(batched)

#dataset = new_dataset

#print(end_to_end_distance)

# Shuffle the dataset
import random
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

random.shuffle(new_dataset)
training_dataset = new_dataset


# Dataset size
print(len(training_dataset))

##
# Encoder Definition
##
class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.mlp1 = nn.Linear(312, 100)
        self.mlp2 = nn.Linear(100, 50)
        self.mlp3 = nn.Linear(50, 31)
        self.mu = nn.Linear(31,31)
        self.log_var = nn.Linear(31,31)

    def forward(self, x):
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = F.relu(self.mlp3(x))
        mu, log_var = torch,sigmoid(self.mu(x)), torch.sigmoid(self.log_var(x))
        return mu, log_var

##
# Generator Definition
##

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.mlp1 = nn.Linear(32,50)
        self.mlp2 = nn.Linear(50,100)
        self.mlp3 = nn.Linear(100,312)

    def forward(self, z):
        z = torch.sigmoid(self.mlp1(z))
        z = torch.sigmoid(self.mlp2(z))
        pred_x = self.mlp3(z)
        return pred_x

    def generate(self, batch_size, max_steps):
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
# Encoder and Generator Initialization
##

encoder = Encoder().cuda()
#generator = Generator().cuda()
decoder = Decoder().cuda()
###
# Discriminator Definition
###

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.mlp1 = nn.Linear(251, 10)
        self.mlp2 = nn.Linear(10,1)

    def forward(self,frame, x):
        x = torch.cat([frame, x],-1)
        x = torch.sigmoid(self.mlp1(x))
        x = torch.sigmoid(self.mlp2(x))
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

    # Functions
    def calc_phi(self,forces_obj, spos, sbox):
        _, _, r12 = self.calculate_distances(
            spos, forces_obj.par.dihedrals[:, [0, 1]], sbox
        )
        _, _, r23 = self.calculate_distances(
            spos, forces_obj.par.dihedrals[:, [1, 2]], sbox
        )
        _, _, r34 = self.calculate_distances(
            spos, forces_obj.par.dihedrals[:, [2, 3]], sbox
        )
        phi_list = self.evaluate_torsion(
            r12, r23, r34, forces_obj.par.dihedral_params
        )
        return phi_list

    def calculate_distances(self,atom_pos, atom_idx, box):
        atom_pos = atom_pos[0]
        direction_vec = self.wrap_dist(atom_pos[atom_idx[:, 0]] - atom_pos[atom_idx[:, 1]], box)
        dist = torch.norm(direction_vec, dim=1)
        direction_unitvec = direction_vec / dist.unsqueeze(1)
        return dist, direction_unitvec, direction_vec


    def wrap_dist(self,dist, box):
        if box is None or torch.all(box == 0):
            wdist = dist
        else:
            wdist = dist - box.unsqueeze(0) * torch.round(dist / box.unsqueeze(0))
        return wdist

                        
    def evaluate_torsion(self,r12, r23, r34, torsion_params, explicit_forces=True):
        # Calculate dihedral angles from vectors
        crossA = torch.cross(r12, r23, dim=1)
        crossB = torch.cross(r23, r34, dim=1)
        crossC = torch.cross(r23, crossA, dim=1)
        normA = torch.norm(crossA, dim=1)
        normB = torch.norm(crossB, dim=1)
        normC = torch.norm(crossC, dim=1)
        normcrossB = crossB / normB.unsqueeze(1)
        cosPhi = torch.sum(crossA * normcrossB, dim=1) / normA
        sinPhi = torch.sum(crossC * normcrossB, dim=1) / normC
        phi = -torch.atan2(sinPhi, cosPhi)
        return phi

    def calc_energy(self, coords, time=None):
        '''Calc energies with torchmd given a set of coordinates'''
        # Reshape array if needed
        if not coords.shape == (self.num_atoms, 3, 1):
            coords = np.reshape(coords, (self.num_atoms, 3, 1))
        # Set positions for system object
        self.system.set_positions(coords)
        # Evaluate current energy and forces. Forces are modified in-place
        forces = Forces(self.parameters, cutoff=9, rfa=True, switch_dist=7.5)
        # Calculate torsion anlges
        my_phis = self.calc_phi(forces, self.system.pos, self.system.box)        
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
            energies = (Epot, my_phis)
        else:
            energies = Epot[0][self.etype]
            energies = [energies]
        return energies

'''
colvar = {
    "name": "E2End Harm",
    "fk": 1.0,
    "cent_0": 12.0,
    "cent_1": 34.0,
    "T": 500000/50,
    "group1": [3],
    "group2": [98]
}
'''
colvar = {
    "name": "E2End Harm",
    "fk": 1.0,
    "cent_0": 12.0,
    "cent_1": 34.0,
    "T": 50000/50,
    "group1": [3],
    "group2": [98]
}
##
# Configurations for Energy Calculation
##
data_dir = "./../../V_Calculations/Test-8_torsion_angle_calc/data/"
psf_file = "full_da-1.3.prmtop"  # prmtop file made using Charmm params with chamber in parmed
parameter_file = "full_da-1.3.prmtop" # contains bonds, angles, dihedrals, electrostatics, lj; no 1-4, impropers or external
##
# Begin Pre-Training
##
import torch.optim as optim
learning_rate = 1e-2

pretrain_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

max_epochs = 5

max_steps = 1002
pretrain_loss = []
import time
print(len(pretrain_dataset))
for epoch in range(max_epochs):

    epoch_pretrain_loss = []
    for t, data in pretrain_dataset:
        # start = time.time()
        alpha = torch.tensor(epoch/max_epochs).float().cuda()
        one = torch.tensor(1.0).float().cuda() 
        bonds_factor = torch.tensor(1.0).float().cuda()
        angle_factor = torch.tensor(1.0).float().cuda()
        dihedral_factor = torch.tensor(1.0).float().cuda()
        x = torch.tensor(data).float().cuda()
        x = x.view(1,104*3)


        #t = torch.tensor(t).view(1,1)
        t = t/max_steps
        #t = t.float().cuda()
        z = torch.normal(0,1,size=(1,31)).cuda()

        z = torch.cat((torch.tensor(t).float().view(1,1).cuda(),z),1)

        pred_x = decoder(z)

        pretrain_optimizer.zero_grad() 
        sys_decal = Energy(data_dir, psf_file, parameter_file, colvar=colvar)  
        t = t*max_steps
        potential, phi = sys_decal.calc_energy(pred_x.view(104,3,1),t)
        _, target_phi = sys_decal.calc_energy(x.view(104,3,1),t)
        # print(target_phi.size())
        phi_loss =  F.mse_loss(phi, target_phi)
        total_pot = torch.zeros(1,1).cuda()
        for key in potential[0].keys():
            if key == 'bonds':
                total_pot += bonds_factor*potential[0][key]
            if key == 'angles':
                total_pot += angle_factor*potential[0][key]
            if key == 'dihedrals':
                total_pot += dihedral_factor*potential[0][key]   
        total_loss = alpha*total_pot + (one - alpha)*phi_loss
        total_loss.backward()
        clipping_value = 1 # arbitrary value of your choosing
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clipping_value)
        epoch_pretrain_loss.append(total_loss.item()) 
        pretrain_optimizer.step()
    #     end = time.time()
    #     print(str(end-start) + 's')
    #     break
    # break
    pretrain_loss.append(np.mean(epoch_pretrain_loss))

decoder.eval()
# Go through the reaction coordinate of the trajectory
max_generation_steps = 20
predictions = []
for t in range(max_generation_steps):
    gen_frame = decoder.generation_step(t, max_generation_steps)
    predictions.append(gen_frame.view(104,3))

predictions = torch.stack(predictions)
predictions = predictions.cpu().detach().numpy()
# Save predictions into VMD format
frame_num = predictions.shape[0]

nAtoms = "104"
outName = "Pretrain.xyz"
with open(outName, "w") as outputfile:
    for frame_idx in range(frame_num):
        
        frame = predictions[frame_idx]
        outputfile.write(str(nAtoms) + "\n")
        outputfile.write(" generated by JK\n")

        atomType = "CA"
        for i in range(104):
            line = str(frame[i][0]) + " " + str(frame[i][1]) + " " + str(frame[i][2]) + " "
            line += "\n"
            outputfile.write("  " + atomType + "\t" + line)

print("=> Finished Pre-Train Generation <=")

# print(str(end-start) + 's')
plt.figure()
plt.plot(range(len(pretrain_loss)), pretrain_loss)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.savefig('pretrain_loss.png')

torch.save(decoder.state_dict(), 'pretrain-decoder.pt')
print('Pretrain Done')

###
# Begin GAN Training
###
d_learning_rate = 1e-3
g_learning_rate = 1e-3

g_optimizer = optim.Adam(decoder.parameters(), lr=g_learning_rate)
d_optimizer = optim.Adam(discriminator.parameters(),lr=d_learning_rate)

# Initalize BCELoss function
criterion = nn.BCELoss()

max_epochs = 3
Ng = 1
Nd = 1
Ni = 1

generator_loss = []
discriminator_loss = []
potential_loss = []

decoder.train()
print(len(training_dataset))
import time
for epoch in range(max_epochs):

    epoch_generator_loss = []
    epoch_discriminator_loss = []
    epoch_potential_loss = []
    for data in training_dataset:
        # start = time.time()
        iteration_generator_loss = []
        iteration_discriminator_loss = []
        iteration_potential_loss = []
        for _ in range(Ng):

            ###
            # (2) Update G Network: maximize log(D(G(z)))
            ###
            g_optimizer.zero_grad()
            # Generator
            t,output,pt = decoder.generate(batch_size,1002)
            output = output.view(batch_size,104*3)
            # D(G(z)))
            phis = []
            for out, p in zip(output,pt):
                sys_decal = Energy(data_dir, psf_file, parameter_file, colvar=colvar)  
                _, phi = sys_decal.calc_energy(out.view(104,3,1),p)
                phis.append(phi.unsqueeze(0))
            phis = torch.cat(phis, 0)
            pred = discriminator(t,phis).squeeze(0)
            label =  torch.ones((batch_size,1)).float().cuda()
            g_fake = criterion(pred, label)
            iteration_generator_loss.append(g_fake.item())
            g_fake.backward()
            # Update generator weights
            g_optimizer.step()

            del t
            del output 
            del sys_decal

        epoch_generator_loss.append(np.mean(iteration_generator_loss))
        for _ in range(Nd):
            ###
            # (1) Update D Network: maximize log(D(x)) + log (1 - D(G(z)))
            ###
            discriminator.zero_grad()
            label = torch.ones((batch_size,1)).float().cuda()
            x = torch.tensor(data).float().cuda()
            t = x[:,:1]
            pt = t.clone().cpu().detach().numpy().tolist()
            x = x[:,1:].view(batch_size,104*3)
            # D(G(z)))
            phis = []
            for out, p in zip(x,pt):
                sys_decal = Energy(data_dir, psf_file, parameter_file, colvar=colvar)  
                _, phi = sys_decal.calc_energy(out.view(104,3,1),p)
                phis.append(phi.unsqueeze(0))
            phis = torch.cat(phis, 0)
            pred = discriminator(t,phis).squeeze(0)
            d_real = criterion(pred, label)
            d_real.backward() 

            # Train with fake examples
            # Generator
            t,output,pt = decoder.generate(batch_size,1002)
            output = output.view(batch_size,104*3)
            # D(G(z)))
            phis = []
            for out, p in zip(output,pt):
                sys_decal = Energy(data_dir, psf_file, parameter_file, colvar=colvar)  
                _, phi = sys_decal.calc_energy(out.view(104,3,1),p)
                phis.append(phi.unsqueeze(0))
            phis = torch.cat(phis, 0)
            pred = discriminator(t,phis).squeeze(0)
            label = torch.zeros((batch_size,1)).float().cuda()
            d_fake = criterion(pred, label)
            iteration_discriminator_loss.append(d_fake.item() + d_real.item())
            d_fake.backward()
            # Update discriminator weights after loss backward from BOTH d_real AND d_fake examples
            d_optimizer.step()

            del x
            del t
            del output
            del sys_decal

        epoch_discriminator_loss.append(np.mean(iteration_discriminator_loss))

        ###
        # (3) Update G Network: minimize log(I(G(z)))
        ###
        bonds_factor = torch.tensor(1.0).float().cuda()
        angle_factor = torch.tensor(1.0).float().cuda()
        dihedral_factor = torch.tensor(1.0).float().cuda()
        improper_factor = torch.tensor(1.0).float().cuda()
        lj_factor = torch.tensor(1.0).float().cuda()
        electrostatics_factor = torch.tensor(1.0).float().cuda()
        repulsion_factor = torch.tensor(1.0).float().cuda()
        end2end_factor = torch.tensor(200.0).float().cuda()
        for i in range(8):
            g_optimizer.zero_grad()
            # Generator
            _,output,t = decoder.generate(1,1002)
            output = output.view(1,312).view(104,3,1)
            # D(G(z)))
            #output = output.detach().cpu()
            sys_decal = Energy(data_dir, psf_file, parameter_file, colvar=colvar)
            potential, _ = sys_decal.calc_energy(output,t)
            total_pot = torch.zeros(1,1).cuda()
            for key in potential[0].keys():
                if key == 'bonds':
                    total_pot += bonds_factor*potential[0][key]
                if key == 'angles':
                    total_pot += angle_factor*potential[0][key]
                if key == 'dihedrals':
                    total_pot += dihedral_factor*potential[0][key]
                if key == 'impropers':
                    total_pot += improper_factor*potential[0][key]
                if key == 'lj':
                    total_pot += lj_factor*potential[0][key]
                if key == 'electrostatics':
                    total_pot += electrostatics_factor*potential[0][key]
                if key == 'repulsion':
                    total_pot += repulsion_factor*potential[0][key]
            # Update generator weights
            total_pot += end2end_factor*potential[0]['E2End Harm']
            iteration_potential_loss.append(total_pot.item())
            total_pot.backward()
            clipping_value = 1 # arbitrary value of your choosing
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), clipping_value)
            g_optimizer.step()

            del sys_decal
            del t, output
            del potential
            del total_pot

        # end = time.time()
        
        del bonds_factor
        del angle_factor
        del dihedral_factor
        del improper_factor
        del lj_factor
        del electrostatics_factor
        del repulsion_factor
        del end2end_factor
        epoch_potential_loss.append(np.mean(iteration_potential_loss))
        # print(str(end-start) + 's')
        # break

    generator_loss.append(np.mean(epoch_generator_loss))
    discriminator_loss.append(np.mean(epoch_discriminator_loss))
    potential_loss.append(np.mean(epoch_potential_loss))
print('Done')

##
# Generation
##
decoder.eval()
# Go through the reaction coordinate of the trajectory
max_generation_steps = 20
predictions = []
for t in range(max_generation_steps):
    gen_frame = decoder.generation_step(t, max_generation_steps)
    predictions.append(gen_frame.view(104,3))

predictions = torch.stack(predictions)
predictions = predictions.cpu().detach().numpy()
# Save predictions into VMD format
frame_num = predictions.shape[0]

nAtoms = "104"
outName = "GAN_8.xyz"
with open(outName, "w") as outputfile:
    for frame_idx in range(frame_num):
        
        frame = predictions[frame_idx]
        outputfile.write(str(nAtoms) + "\n")
        outputfile.write(" generated by JK\n")

        atomType = "CA"
        for i in range(104):
            line = str(frame[i][0]) + " " + str(frame[i][1]) + " " + str(frame[i][2]) + " "
            line += "\n"
            outputfile.write("  " + atomType + "\t" + line)

print("=> Finished Generation <=")

torch.save(decoder.state_dict(), 'decoder-gan.pt')
##
# Plot Loss
##
plt.figure()
plt.plot(range(len(discriminator_loss)), discriminator_loss)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.savefig('discriminator_loss.png')

plt.figure()
plt.plot(range(len(generator_loss)), generator_loss)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.savefig('generator_loss.png')

plt.figure()
plt.plot(range(len(potential_loss)), potential_loss)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.savefig('potential_loss.png')

print('Done Done')
#!/usr/bin/env python
'''
##
# Molecular cGAN for Min-Action Pathways
##
##############################################################
# Author:               John Cava, John Vant
# Email:              ?????, jvant@asu.edu
# Affiliation:   ASU Biodesign Institute
# Date Created:          211110
##############################################################
# Usage:
##############################################################
# Notes:
##############################################################
'''
###
# Import
###
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from moleculekit.molecule import Molecule
import os
from torchmd.forcefields.forcefield import ForceField
from torchmd.parameters import Parameters
# from torchmd.integrator import maxwell_boltzmann
from torchmd.systems import System
from torchmd.forces import Forces
import torch.optim as optim
import time


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
        self.d1 = nn.Dropout(p=0.2)
        self.mlp2 = nn.Linear(50,75)
        self.d2 = nn.Dropout(p=0.4)
        self.mlp3 = nn.Linear(75,312)

    def forward(self, z):
        z = torch.sigmoid(self.mlp1(z))
        #z = self.d1(z)
        z = torch.sigmoid(self.mlp2(z))
        #z = self.d2(z)
        pred_x = self.mlp3(z)
        return pred_x

    def generate(self, batch_size, max_steps):
        t = random.choices(range(max_steps),k=batch_size)
        picked_t = t
        t = torch.tensor(t).view(batch_size,1)
        t = t/max_steps
        t = t.float().cuda()
        z = torch.normal(0,1,size=(batch_size,32)).cuda()
        #z = torch.cat((t,z),1)
        z = torch.sigmoid(self.mlp1(z))
        z = torch.sigmoid(self.mlp2(z))
        z = self.mlp3(z)
        return t, z, picked_t

    def generation_step(self, t, max_steps):
        t = torch.tensor(t).view(1,1)
        t = t/max_steps 
        t = t.float().cuda()
        z = torch.normal(0,1,size=(1,32)).cuda()
        #z = torch.cat((t,z),1)
        z = torch.sigmoid(self.mlp1(z))
        z = torch.sigmoid(self.mlp2(z))
        z = self.mlp3(z)
        return z


###
# Discriminator Definition
###


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.mlp1 = nn.Linear(251, 10)
        self.mlp2 = nn.Linear(10,1)

    def forward(self, frame, x):
        x = torch.cat([frame, x],-1)
        # x = frame
        x = torch.sigmoid(self.mlp1(x))
        x = torch.sigmoid(self.mlp2(x))
        return x


###
# (v) Potential Loss Function
###

# Define Class
class Energy:
    UNITS = "kcal/mol"

    def __init__(self, psf_file, parameter_file,
                 colvar=None, device="cuda", precision=torch.float, etype='all'):
        self.etype = etype
        # Make Molecule object
        mol = Molecule(psf_file)  # Reading the system topology
        self.num_atoms = mol.numAtoms
        # Create Force Field object
        ff = ForceField.create(mol, parameter_file)
        self.dtype = torch.float
        self.device = torch.device("cuda")
        self.parameters = Parameters(ff, mol, precision=precision, device=self.device)
        # Convert Moleculekit Molecule object to torchmd system object
        self.system = System(self.num_atoms, nreplicas=1, precision=precision, device=self.device)
        if not colvar is None:
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

    def wrap_dist(self, dist, box):
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
        if not self.colvar_name is None:
            if time == None:
                print("No time provided.  Exiting calculation.")
                exit()
            cur_center = ((self.colvar_cent_1-self.colvar_cent_0)/self.colvar_T)*time + self.colvar_cent_0
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


# Make Custom Dataset Object
class SystemDataset(torch.utils.data.Dataset):

    def __init__(self, psf_file, parameter_file, traj_files, colvar, transform=None, stride=10):
        self.psf_file = psf_file
        self.parameter_file = parameter_file
        self.traj_files = traj_files
        self.transform = transform
        self.colvar = colvar
        self.sys_energy = Energy(self.psf_file, self.parameter_file, colvar=self.colvar, device="cuda")
        self.num_atoms = self.sys_energy.num_atoms
        print("=> Start loading data <=")
        for file_ in self.traj_files:
            X = np.load(file_)
            X = X[::stride]
            print(f"Trajectory length of file {os.path.basename(file_)} after stride: {len(X)}")
            dataset = []
            append = dataset.append
            for frame_num in range(X.shape[0]):
                append((frame_num, X[frame_num,:,:]))
        self.data = dataset

                    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        frame_num, frame = sample
        frame_num = torch.tensor(frame_num).float().cuda()
        frame = torch.tensor(frame).float().cuda()
        potential, phi = self.sys_energy.calc_energy(frame.view(self.num_atoms,3,1),frame_num)
        sample = {'frame': frame, 'bonds': potential[0]['bonds'],
                                  'angles': potential[0]['angles'],
                                  'dihedrals': potential[0]['dihedrals'],
                                  'impropers': potential[0]['impropers'],
                                  'lj': potential[0]['lj'],
                                  'electrostatics': potential[0]['electrostatics'],
                                  'End2End': potential[0]['E2End Harm'],
                                  'potentials': potential[0],
                                  'phi': phi}

        if self.transform:
            sample = self.transform(sample)

        return sample, frame_num



###
# RMSD Calculation
###

# following code from https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8 (Author: Guillaume Bourvier) 
def find_rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
        >>> A = torch.tensor([[1., 1.], [2., 2.], [1.5, 3.]], dtype=torch.float)
        >>> R0 = torch.tensor([[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]], dtype=torch.float)
        >>> B = (R0.mm(A.T)).T
        >>> t0 = torch.tensor([3., 3.])
        >>> B += t0
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
        >>> B *= torch.tensor([-1., 1.])
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()


# Take find_rigid_alignment() and use it on a batch prediction and target
def rmsd(pred, target):

    total_rmsd = torch.zeros(1,1).float().cuda()

    for p,t in zip(pred, target):

        R,t = find_rigid_alignment(p, t)
        p_aligned = (R.mm(p.T)).T + t
        rmsd = torch.sqrt(((p_aligned - t)**2).sum(axis=1).mean())
        total_rmsd += rmsd

    return total_rmsd


def save_data_xyz(frames, outfileName):
    frames = torch.stack(frames)
    frames = frames.cpu().detach().numpy()
    # Save frames into VMD format
    nAtoms = frames.shape[1]
    with open(outfileName, "w") as outputfile:
        for data in frames:
            frame = data
            outputfile.write(str(nAtoms) + "\n")
            outputfile.write(" generated by JK\n")
            atomType = "CA"
            for i in range(nAtoms):
                line = str(frame[i][0]) + " " + str(frame[i][1]) + " " + str(frame[i][2]) + " "
                line += "\n"
                outputfile.write("  " + atomType + "\t" + line)


###
# Important Variables and Settings
###
input_size = 3
hidden_size = 128
history_size = 15
lead_time = 2
M = 5
num_layers = 1
batch_size = 32
seed = 666
# Data config
stride = 10
data_files_regex = './../../All_ML_Training_Data/210905_SMD_decaalanine/SMD/output/rand_orein_npy_traj/*npy'
# Configurations for Energy Calculation
data_dir = "./../../V_Calculations/Test-6_full_system/data/"
psf_file = "full_da-1.3.prmtop"  # This is a special psf file with improper connectivity deleted
parameter_file = "full_da-1.3.prmtop" # bond, angles, dihedrals, electrostatics, lj; no 1-4, impropers or external
colvar = {
    "name": "E2End Harm",
    "fk": 1.0,
    "cent_0": 12.0,
    "cent_1": 34.0,
    "T": 50000/50,
    "group1": [3],
    "group2": [98]
}

# plotting
matplotlib.use('Agg')

###
# Initialize Dataset
###
psf_file = os.path.join(data_dir, psf_file)
parameter_file = os.path.join(data_dir, parameter_file)
traj_files = glob.glob(data_files_regex)
training_dataset = SystemDataset(psf_file, parameter_file, traj_files, colvar)


##
# Encoder, Generator and Discriminator Initialization
##
print("=> Start NN Initialization <=")
# Seeding randomnes
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
encoder = Encoder().cuda()
decoder = Decoder().cuda()
discriminator = Discriminator().cuda()


###
# Begin Pre-Training
###
# pretrain_dataset = SystemDataset(pretrain_dataset, colvar)
# pretrain_dataloader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=batch_size,
#                         shuffle=True, num_workers=0)

###
# Important pretrain Variables and Settings
###
# V_intro_epoch = 20
# max_epochs = 30
# dist_decline_epoch = 26
# max_steps = 1001
# pretrain_loss = []


# pred_sys = Energy(data_dir, psf_file, parameter_file, colvar=colvar)

# predictions = []
# q = 0
# learning_rate = 1e-2
# print(f"\n===> Starting {q}, learning rate {learning_rate} <===\n")
# pretrain_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
# for epoch in range(max_epochs):
#     if epoch == V_intro_epoch + 1:
#         pretrain_optimizer = optim.Adam(decoder.parameters(), lr=1e-4)
#     start = time.time()
#     epoch_pretrain_loss = []
#     for i, data in enumerate(pretrain_dataloader):

#         data, t = data
#         if not epoch > V_intro_epoch:
#             alpha = torch.tensor(0.0).float().cuda()
#         else:
#             alpha = torch.tensor((epoch-V_intro_epoch)/(max_epochs-V_intro_epoch-1)).float().cuda()

#         if not epoch > dist_decline_epoch:
#             alpha2 = torch.tensor(0.0).float().cuda()
#         else:
#             alpha2 = torch.tensor((epoch-dist_decline_epoch)/(max_epochs-dist_decline_epoch-1)).float().cuda()

#         one = torch.tensor(1.0).float().cuda()
#         bonds_factor = torch.tensor(1.0).float().cuda()
#         angle_factor = torch.tensor(1.0).float().cuda()
#         dihedral_factor = torch.tensor(1.0).float().cuda()
#         improper_factor = torch.tensor(1.0).float().cuda()
#         lj_factor = torch.tensor(1.0).float().cuda()
#         electrostatics_factor = torch.tensor(1.0).float().cuda()
#         dis_factor = torch.tensor(1.0).float().cuda()

#         x = data['frame']
#         bsize = x.size()[0]
#         x = x.view(bsize, 104*3)
#         target_phi = data['phi']
#         z = torch.normal(0,1,size=(bsize,32)).cuda()
#         t = t.unsqueeze(-1).float().cuda()/max_steps
#         #z = torch.cat((t,z),1)

#         pred_x = decoder(z)

#         pretrain_optimizer.zero_grad()
#         total_pot = torch.zeros(1,1).cuda()
#         total_phi = torch.zeros(1,1).cuda()
#         count=0
#         for px, t_phi, pt in zip(pred_x, target_phi, t):
#             if count % int(batch_size/2) == 0:
#                 predictions.append(px.view(104,3))
#             count+=1
#             potential, phi = pred_sys.calc_energy(px.view(104,3,1), pt)
#             # print(f"=============t_phi {phi.shape}\n", t_phi)
#             # print(f"=============phi {phi.shape}\n", phi)
#             # total_phi +=  F.mse_loss(phi, t_phi)
#             for key in potential[0].keys():
#                 if key == 'bonds':
#                     total_pot += bonds_factor*potential[0][key]/batch_size
#                 if key == 'angles':
#                     total_pot += angle_factor*potential[0][key]/batch_size
#                 if key == 'dihedrals':
#                     total_pot += dihedral_factor*potential[0][key]/batch_size
#                 if key == 'impropers':
#                     total_pot += improper_factor*potential[0][key]/batch_size
#                 # if q > 1:
#                 if key == 'lj':
#                     total_pot += lj_factor*potential[0][key]/batch_size
#                 if key == 'electrostatics':
#                     total_pot += electrostatics_factor*potential[0][key]/batch_size
#                 # if torch.isinf(total_pot):
#                 #     print(f"\nGot Inf on key {key}\n")
#         dist_pred = torch.cdist(pred_x.view(bsize,104,3),pred_x.view(bsize,104,3))
#         dist_target = torch.cdist(x.view(bsize,104,3),x.view(bsize,104,3))
#         recon_loss = dis_factor*F.mse_loss(dist_pred, dist_target)
#         # total_loss = alpha*total_pot + (one - alpha)*recon_loss*1e-3
#         # print(total_pot)
#         # rmsd_loss = rmsd(pred_x.view(bsize,104,3),x.view(bsize,104,3))
#         print(f"alpha:{alpha}\nalpha2:{alpha2}\nPot:{total_pot}")
#         total_loss = alpha*total_pot + (1-alpha2)*recon_loss*1e-3
#         total_loss.backward()
#         clipping_value = 1 # arbitrary value of your choosing
#         torch.nn.utils.clip_grad_norm_(decoder.parameters(), clipping_value)
#         epoch_pretrain_loss.append(total_loss.item())
#         pretrain_optimizer.step()
#         end = time.time()
#     l = np.mean(epoch_pretrain_loss)
#     # if l < min_val:
#     #     min_val = l
#     #     torch.save(decoder.state_dict(), 'pretrain-decoder.pt')
#     pretrain_loss.append(l)
#     print(f"\n===> Learning Iter {q}, Finished Epoch {epoch} in {end-start:.2f} s <===\n")
#     print(f"Mean epoch pretrain loss: {l}")
#     print(f"Pre-Training is {(q*max_epochs + epoch)/(max_epochs*8)*100:.4f} % complete")

# torch.save(decoder.state_dict(), 'pretrain-decoder.pt')
# plt.figure()
# plt.plot(range(len(pretrain_loss)), pretrain_loss)
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.savefig('pretrain_loss.png')

decoder.load_state_dict(torch.load('pretrain-decoder.pt'))
decoder.eval()
# Go through the reaction coordinate of the trajectory
max_generation_steps = 20
predictions = []
for t in range(max_generation_steps):
    gen_frame = decoder.generation_step(t, max_generation_steps)
    predictions.append(gen_frame.view(104,3))

exit("Generater is working?")
# save_data_xyz(predictions, "pretrain_generated.xyz")
# print("=> Finished Pre-Train Generation <=")

# Save model
# torch.save(decoder.state_dict(), 'pretrain-decoder.pt')
# print('Pretrain Done')
# exit()


###
# Begin GAN Training
###
decoder.load_state_dict(torch.load('./pretrain-decoder.pt'))
decoder.eval()
# Important GAN parameters
nAtoms = "104"
d_learning_rate = 1e-3
g_learning_rate = 1e-3
g_optimizer = optim.Adam(decoder.parameters(), lr=g_learning_rate)
d_optimizer = optim.Adam(discriminator.parameters(),lr=d_learning_rate)

# Initalize BCELoss function
criterion = nn.BCELoss()

max_epochs = 10
Ng = 1
Nd = 1
Ni = 1

generator_loss = []
discriminator_loss = []
potential_loss = []

decoder.train()
print(len(training_dataset))

# Dataloader

training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)


for epoch in range(max_epochs):
    epoch_generator_loss = []
    epoch_discriminator_loss = []
    epoch_potential_loss = []
    start = time.time()
    for i, data in enumerate(training_dataloader):
        batch_size = len(data[1])
        iteration_generator_loss = []
        iteration_discriminator_loss = []
        iteration_potential_loss = []
        for _ in range(Ng):
            ###
            # (2) Update G Network: maximize log(D(G(z)))
            ###
            g_optimizer.zero_grad()
            # Generator
            t, output, pt = decoder.generate(batch_size,1001)
            output = output.view(batch_size,104*3)
            # D(G(z)))
            phis = []
            append = phis.append
            for out, p in zip(output,pt):
                pred_sys = Energy(psf_file, parameter_file, colvar=colvar)
                _, phi = pred_sys.calc_energy(out.view(104,3,1),p)
                append(torch.unsqueeze(phi,0))
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
            del pred_sys

        epoch_generator_loss.append(np.mean(iteration_generator_loss))
        for _ in range(Nd):
            ###
            # (1) Update D Network: maximize log(D(x)) + log (1 - D(G(z)))
            ###
            discriminator.zero_grad()
            label = torch.ones((batch_size,1)).float().cuda()
            data, t_real = data
            # x = torch.tensor(data['frame']).float().cuda()
            t_real = t_real.view(batch_size,1)
            pt = t_real
            x = data['frame']
            x = x.view(batch_size,104*3)
            # D(G(z)))
            phis = []
            append = phis.append
            for out, p in zip(x,pt):
                pred_sys = Energy(psf_file, parameter_file, colvar=colvar)
                _, phi = pred_sys.calc_energy(out.view(104,3,1),p)
                append(torch.unsqueeze(phi, 0))
            phis = torch.cat(phis, 0)
            pred = discriminator(t_real, phis).squeeze(0)
            d_real = criterion(pred, label)
            d_real.backward() 

            # Train with fake examples
            # Generator
            t,output,pt = decoder.generate(batch_size,1001)
            output = output.view(batch_size,104*3)
            # D(G(z)))
            phis = []
            append = phis.append
            for out, p in zip(output,pt):
                pred_sys = Energy(psf_file, parameter_file, colvar=colvar)
                _, phi = pred_sys.calc_energy(out.view(104,3,1),p)
                append(torch.unsqueeze(phi,0))
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
            del pred_sys

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
        for i in range(Ni):
            g_optimizer.zero_grad()
            # Generator
            _,output,t = decoder.generate(batch_size,1001)
            # output = output.view(batch_size,104*3).view(104,3,1)
            output = output.view(batch_size,104*3).view(batch_size,104,3,1)  # JK not sure what this was about
            # D(G(z)))
            #output = output.detach().cpu()
            total_pot = torch.zeros(1,1).cuda()
            for out, pt in zip(output,t):
                pred_sys = Energy(psf_file, parameter_file, colvar=colvar)
                potential, _ = pred_sys.calc_energy(out,pt)
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
            total_pot = total_pot/batch_size
            print(total_pot)
            iteration_potential_loss.append(total_pot.item())
            total_pot.backward()
            clipping_value = 1 # arbitrary value of your choosing
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), clipping_value)
            g_optimizer.step()

            del pred_sys
            del t, output
            del potential
            del total_pot

        del bonds_factor
        del angle_factor
        del dihedral_factor
        del improper_factor
        del lj_factor
        del electrostatics_factor
        del repulsion_factor
        del end2end_factor
        epoch_potential_loss.append(np.mean(iteration_potential_loss))

        # break
    print(f"Training is {100*epoch+1/max_epochs}% complete\nEpoch {epoch} took {time.time()-start} s")
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

save_data_xyz(predictions, "cGAN_generated.xyz")
print("=> Finished Generation <=")

# Save model
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

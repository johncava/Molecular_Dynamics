#!/usr/bin/env python
'''
##
# Molecular cGAN for Min-Action Pathways
##
##############################################################
# Author:               John Cava, John Vant
# Email:                jcava@asu.edu, jvant@asu.edu
# Affiliation:        ASU Biodesign Institute
# Date Created:               211110
##############################################################
# Usage:
##############################################################
# Notes: in update_D() true_label_smooth and fake_label_smooth are added
# in order to avoid the discriminator in being overconfident in predicting true and fake as 1 and 0
# respectively. Instead it is of a 'smoothed' value such as 0.9 and 0.3 respectively.
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

# plotting
matplotlib.use('Agg')


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
        z = torch.sigmoid(self.mlp2(z))
        pred_x = self.mlp3(z)
        return pred_x

    def generate(self, batch_size, max_steps):
        t = random.choices(range(max_steps), k=batch_size)
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


###
# Discriminator Definition
###


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()        
        self.mlp1 = nn.Linear(313, 100)
        self.mlp2 = nn.Linear(100,10)
        self.mlp3 = nn.Linear(10,1)

    def forward(self, frame, x):
        x = torch.cat([frame, x],-1)
        # x = frame
        x = torch.sigmoid(self.mlp1(x))
        x = torch.sigmoid(self.mlp2(x))
        x = torch.sigmoid(self.mlp3(x))
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

    def calc_colvar(self, time):
        '''Need to update coords before running this '''
        cur_center = (self.colvar_cent_1-self.colvar_cent_0)*time + self.colvar_cent_0
        grp1_com = self.system.pos[0][self.colvar_group1[0]]
        grp2_com = self.system.pos[0][self.colvar_group2[0]]
        dist = torch.pow(torch.sum(torch.pow(torch.sub(grp2_com, grp1_com),2)),0.5)
        engy = torch.mul(torch.mul(torch.pow(torch.sub(cur_center, dist),2), self.colvar_fk),0.5)
        # force = torch.mul(torch.sub(cur_center, dist), self.colvar_fk)
        return dist, engy

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
            _, engy = self.calc_colvar(time)
            Epot = Epot[0]
            Epot[self.colvar_name] = engy.view(1)
            Epot = [Epot]
        if self.etype == 'all':
            energies = (Epot, my_phis)
        else:
            energies = Epot[0][self.etype]
            energies = [energies]
        return energies

# Define Colvar
colvar = {
    "name": "E2End Harm",
    "fk": 1.0,
    "cent_0": 12.0,
    "cent_1": 34.0,
    "T": 50000/50,
    "group1": [3],
    "group2": [98]
}

# Make Custom Dataset Object
class SystemDataset(torch.utils.data.Dataset):
    '''An extented pytorch dataset object to house molecular trajectory data'''
    def __init__(self, psf_file, parameter_file, traj_files, colvar, pretrain=False, transform=None, stride=10):
        self.psf_file = psf_file
        self.parameter_file = parameter_file
        self.traj_files = traj_files
        self.transform = transform
        self.colvar = colvar
        self.sys_energy = Energy(self.psf_file, self.parameter_file, colvar=self.colvar, device="cuda")
        self.num_atoms = self.sys_energy.num_atoms
        self.pretrain = pretrain
        print("=> Start loading data <=")
        dataset = []
        for file_ in self.traj_files:
            X_data = np.load(file_)
            if self.pretrain == True:
                X = X_data[5:25]
            else:
                X = X_data[::stride]
            print(f"Trajectory length of file {os.path.basename(file_)} after stride: {len(X)}")            
            append = dataset.append
            for frame_num in range(X.shape[0]):
                append((frame_num, X[frame_num,:,:]))
        self.data = dataset
        
        self.num_frames = X_data[::stride].shape[0]
        print("frame num", X_data[::stride].shape[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        frame_num, frame = sample
        frame_num = torch.tensor(frame_num/self.num_frames).float().cuda()
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
# GAN functions
###
def update_pretrain(pretrain_optimizer, data, decoder, V_intro_epoch, dist_decline_epoch, V_only_epoch):
    data, t = data
    if not epoch > V_intro_epoch:
        alpha = torch.tensor(0.0).float().cuda()
    else:
        alpha = torch.tensor((epoch-V_intro_epoch)/(V_only_epoch-V_intro_epoch-1)).float().cuda()

    if not epoch > dist_decline_epoch:
        alpha2 = torch.tensor(0.0).float().cuda()
    else:
        alpha2 = torch.tensor((epoch-dist_decline_epoch)/(V_only_epoch-dist_decline_epoch-1)).float().cuda()
    if epoch >= V_only_epoch:
        alpha = torch.tensor(1.0).float().cuda()
        alpha2 = torch.tensor(1.0).float().cuda()
        
    one = torch.tensor(1.0).float().cuda()
    dis_factor = torch.tensor(1.0).float().cuda()
    x = data['frame']
    x = x.view(bs, num_atoms*3)
    # target_phi = data['phi']
    z = torch.normal(0,1,size=(bs,31)).cuda()
    t = t.unsqueeze(-1).float().cuda()/num_frames
    z = torch.cat((t,z),1)
    pred_x = decoder(z)

    pretrain_optimizer.zero_grad()
    total_pot = torch.zeros(1,1).cuda()
    count=0
    for px,  pt in zip(pred_x, t):
        if count % int(bs/2) == 0:
            pretrain_training_data.append(px.view(num_atoms,3))
        count+=1
        pred_sys = Energy(psf_file, parameter_file, colvar=colvar)
        potential, my_phis = pred_sys.calc_energy(px.view(num_atoms,3,1), pt)
        total_pot += torch.div(torch.sum(torch.stack(
            [potential[0][key] for key in potential[0].keys() if not key == 'E2End Harm'])),bs)
        total_pot += torch.div(torch.tensor(200.0).float().cuda()*potential[0]['E2End Harm'], bs)
        del px, potential, pt, my_phis, pred_sys
    dist_pred = torch.cdist(pred_x.view(bs,num_atoms,3),pred_x.view(bs,num_atoms,3))
    dist_target = torch.cdist(x.view(bs,num_atoms,3),x.view(bs,num_atoms,3))
    recon_loss = dis_factor*F.mse_loss(dist_pred, dist_target)
    print(f"alpha:{alpha}\nalpha2:{alpha2}\nPot:{total_pot}\nRecon loss is {recon_loss}")
    total_loss = alpha*total_pot + (one-alpha2)*recon_loss*1e-3
    total_loss.backward()
    clipping_value = 1 # arbitrary value of your choosing
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clipping_value)
    del total_pot, pred_x, recon_loss, dist_target, dist_pred, z, x,data, t
    return total_loss



def update_G(decoder, discriminator):
    '''
    # (2) Update G Network: maximize log(D(G(z)))
    '''
    g_optimizer.zero_grad()
    # Generator
    t, output, fn = decoder.generate(bs,num_frames)
    output = output.view(bs,num_atoms*3)
    # D(G(z)))
    pred = discriminator(t, output).squeeze(0)
    label =  torch.ones((bs,1)).float().cuda()
    g_fake = criterion(pred, label)
    iteration_generator_loss.append(g_fake.item())
    g_fake.backward()
    # Update generator weights
    g_optimizer.step()

    del t, fn, output, label, pred


def update_D(data, discriminator, decoder):
    '''
    # (1) Update D Network: maximize log(D(x)) + log (1 - D(G(z)))
    '''
    true_label_smooth = torch.tensor(0.9).float().cuda()
    discriminator.zero_grad()
    label = torch.ones((bs,1)).float().cuda()
    label = true_label_smooth*label
    data, t_real = data
    rt = t_real.view(bs,1)
    x = data['frame']
    x = x.view(bs,num_atoms*3)

    # D(G(z)))
    pred = discriminator(rt, x).squeeze(0)
    d_real = criterion(pred, label)
    d_real.backward() 

    # Train with fake examples
    # Generator
    t, output, fn = decoder.generate(bs,num_frames)
    output = output.view(bs,num_atoms*3)
    # D(G(z)))
    pred = discriminator(t,output).squeeze(0)
    fake_label_smooth = torch.tensor(0.3).float().cuda()
    label = torch.zeros((bs,1)).float().cuda()
    label = fake_label_smooth*label
    d_fake = criterion(pred, label)
    iteration_discriminator_loss.append(d_fake.item() + d_real.item())
    d_fake.backward()
    # Update discriminator weights after loss backward from BOTH d_real AND d_fake examples
    d_optimizer.step()

    del t, rt, t_real, x, output, label, pred, data


def update_G_net(decoder):
    '''
    # (3) Update G Network: minimize log(I(G(z)))
    '''
    bs_gen = 8
    g_optimizer.zero_grad()
    # Generator
    pt, output, fn = decoder.generate(bs_gen, num_frames)
    output = output.view(bs_gen,num_atoms*3).view(bs_gen,num_atoms,3,1)
    # D(G(z)))
    total_pot = torch.zeros(1,1).cuda()
    for out, t in zip(output, pt):
        pred_sys = Energy(psf_file, parameter_file, colvar=colvar)
        potential, my_phis = pred_sys.calc_energy(out,t)
        # print(t, potential[0]['E2End Harm'])
        total_pot += torch.div(torch.sum(torch.stack(
            [potential[0][key] for key in potential[0].keys() if not key == 'E2End Harm'])),bs_gen)
        total_pot += torch.div(torch.tensor(200.0).float().cuda()*potential[0]['E2End Harm'], bs_gen)
        del out, potential, t, my_phis, pred_sys
    print("Generator Potential", total_pot.item())
    iteration_potential_loss.append(total_pot.item())
    total_pot.backward()
    clipping_value = 1 # arbitrary value of your choosing
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clipping_value)
    g_optimizer.step()

    del pt, output, total_pot, fn


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
# END FUNCTIONS
###

    
###
# Important Variables and Settings
###
batch_size = 32
seed = 666
# Data config
stride = 10
data_files_regex = './../../All_ML_Training_Data/210905_SMD_decaalanine/SMD/output/rand_orein_npy_traj/*npy'
# Configurations for Energy Calculation
data_dir = "./../../V_Calculations/Test-6_full_system/data/"
psf_file = "full_da-1.3.prmtop"  # This is a special psf file with improper connectivity deleted
parameter_file = "full_da-1.3.prmtop" # bond, angles, dihedrals, electrostatics, lj; no 1-4, impropers or external
# colvar = {
#     "name": "E2End Harm",
#     "fk": 1.0,
#     "cent_0": 12.0,
#     "cent_1": 34.0,
#     "T": 50000/50,
#     "group1": [3],
#     "group2": [98]
# }
###
# Important pretrain Variables and Settings
###
V_intro_epoch = 40
max_epochs_pre = 50
dist_decline_epoch = 43
V_only_epoch = 47

###
# Important GAN parameters
###
d_learning_rate = 1e-3
g_learning_rate = 1e-3

###
# Initalize BCELoss function
###
criterion = nn.BCELoss()
max_epochs_GAN = 1
Ng = 1
Nd = 1
Ni = 1


###
# Initialize Dataset
###
psf_file = os.path.join(data_dir, psf_file)
parameter_file = os.path.join(data_dir, parameter_file)
traj_files = glob.glob(data_files_regex)
training_dataset = SystemDataset(psf_file, parameter_file, traj_files, colvar)
num_atoms = training_dataset.num_atoms
num_frames = training_dataset.num_frames

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
decoder.train()
pretrain_dataset = SystemDataset(psf_file, parameter_file, traj_files, colvar, pretrain=True)
pretrain_dataloader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
print("Pre-Training Dataset size:", len(pretrain_dataset))
learning_rate = 1e-2
pretrain_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
pretrain_loss = []
pretrain_training_data = []
for epoch in range(max_epochs_pre):
    print(f"\n===> Starting pre-train epoch {epoch} <===")
    if epoch == V_intro_epoch + 1:
        pretrain_optimizer = optim.Adam(decoder.parameters(), lr=1e-4)
    start = time.time()
    epoch_pretrain_loss = []
    for i, data in enumerate(pretrain_dataloader):
        bs = len(data[1])
        total_loss = update_pretrain(pretrain_optimizer, data, decoder, V_intro_epoch, dist_decline_epoch, V_only_epoch)
        epoch_pretrain_loss.append(total_loss.item())
        pretrain_optimizer.step()
        end = time.time()
        del data
    torch.cuda.empty_cache()
    l = np.mean(epoch_pretrain_loss)
    pretrain_loss.append(l)
    print(f"\n===> Finished Epoch {epoch} in {end-start:.2f} s <===\n")
    print(f"Mean epoch pretrain loss: {l}")
    print(f"Pre-Training is {(1+epoch)/(max_epochs_pre)*100:.4f} % complete")

save_data_xyz(pretrain_training_data, "pretrain_training_data.xyz")
###
# Save state dict and generate and plot loss
###
# Save model
torch.save(decoder.state_dict(), 'pretrain-decoder.pt')

del pretrain_dataset, pretrain_dataloader 

# Plot
plt.figure()
plt.plot(range(len(pretrain_loss)), pretrain_loss)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.yscale("log")
plt.savefig('pretrain_loss.png')

# load model and set to eval mode
decoder.load_state_dict(torch.load('pretrain-decoder.pt'))
decoder.eval()
# Go through the reaction coordinate of the trajectory
max_generation_steps = 20
predictions = []
for t in range(max_generation_steps):
    gen_frame = decoder.generation_step(t, max_generation_steps)
    predictions.append(gen_frame.view(num_atoms,3))

save_data_xyz(predictions, "pretrain_generated.xyz")

del predictions
print("=> Finished Pre-Train Generation <=")

print('Pretrain Done')


###
# Begin GAN Training
###
decoder.load_state_dict(torch.load('./pretrain-decoder.pt'))
# decoder.eval()
decoder.train()
print("Training Dataset size:", len(training_dataset))

generator_loss = []
discriminator_loss = []
potential_loss = []

# Dataloader
training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
g_optimizer = optim.Adam(decoder.parameters(), lr=g_learning_rate)
d_optimizer = optim.Adam(discriminator.parameters(),lr=d_learning_rate)
for epoch in range(max_epochs_GAN):
    epoch_generator_loss = []
    epoch_discriminator_loss = []
    epoch_potential_loss = []
    start = time.time()
    for i, data in enumerate(training_dataloader):
        bs = len(data[1])
        iteration_generator_loss = []
        iteration_discriminator_loss = []
        iteration_potential_loss = []
        for _ in range(Ng):
            print("update_G")
            update_G(decoder, discriminator)

        epoch_generator_loss.append(np.mean(iteration_generator_loss))
        for _ in range(Nd):
            print("update_D")
            update_D(data, discriminator, decoder)

        epoch_discriminator_loss.append(np.mean(iteration_discriminator_loss))

        if i % 10 == 0:
            for _ in range(Ni):
                print("update_G_net")
                update_G_net(decoder)

        epoch_potential_loss.append(np.mean(iteration_potential_loss))
        del data
    print(f"Training is {100*(epoch+1)/max_epochs_GAN}% complete\nEpoch {epoch} took {time.time()-start} s")
    generator_loss.append(np.mean(epoch_generator_loss))
    discriminator_loss.append(np.mean(epoch_discriminator_loss))
    potential_loss.append(np.mean(epoch_potential_loss))
    torch.cuda.empty_cache()
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
    predictions.append(gen_frame.view(num_atoms,3))

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

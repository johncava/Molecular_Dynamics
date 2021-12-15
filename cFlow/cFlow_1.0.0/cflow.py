#!/usr/bin/env python
'''
##
# Molecular cFlow for Min-Action Pathways
##
##############################################################
# Author:               John Cava, John Vant
# Email:                jcava@asu.edu, jvant@asu.edu
# Affiliation:        ASU Biodesign Institute
# Date Created:               211110
##############################################################
# Usage: V1.0.0
##############################################################
# Notes: Conditional Normalizing Flow to generate structure conditioned on a specific frame num
##############################################################
# References:
# cFLOW: https://github.com/kamenbliznashki/normalizing_flows/blob/97a73a01bcee3ac8015dfeeb3cffb035ec39a1f2/maf.py#L590
# Normalize a vector between (0,1): https://stackoverflow.com/questions/19299155/normalize-a-vector-of-3d-coordinates-to-be-in-between-0-and-1/19301193
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
import torch.distributions as D
from torch import distributions
from torch.nn.parameter import Parameter
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
import copy
# plotting
matplotlib.use('Agg')

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
        self.x_values = []
        self.y_values = []
        self.z_values = []
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
                self.x_values += X[frame_num,:,0].tolist()
                self.y_values += X[frame_num,:,1].tolist()
                self.z_values += X[frame_num,:,2].tolist()
        # Normalize the points between (0,1)
        self.max_x, self.min_x = max(self.x_values), min(self.x_values)
        self.max_y, self.min_y = max(self.y_values), min(self.y_values)
        self.max_z, self.min_z = max(self.z_values), min(self.z_values)

        self.scaled_x = 1.0/(self.max_x - self.min_x)
        self.scaled_y = 1.0/(self.max_y - self.min_y)
        self.scaled_z = 1.0/(self.max_z - self.min_z)
        self.scaled_unit = torch.tensor([self.scaled_x, self.scaled_y, self.scaled_z]).float().cuda()
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
        # Scale frame
        frame = frame * self.scaled_unit
        # potential, phi = self.sys_energy.calc_energy(frame.view(self.num_atoms,3,1),frame_num)
        # sample = {'frame': frame, 'bonds': potential[0]['bonds'],
        #                           'angles': potential[0]['angles'],
        #                           'dihedrals': potential[0]['dihedrals'],
        #                           'impropers': potential[0]['impropers'],
        #                           'lj': potential[0]['lj'],
        #                           'electrostatics': potential[0]['electrostatics'],
        #                           'End2End': potential[0]['E2End Harm'],
        #                           'potentials': potential[0],
        #                           'phi': phi}

        # if self.transform:
        #     sample = self.transform(sample)

        # return sample, frame_num
        return frame, frame_num

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
# cFlow Definitions
# Code From: https://github.com/kamenbliznashki/normalizing_flows/blob/97a73a01bcee3ac8015dfeeb3cffb035ec39a1f2/maf.py#L590
###
class LinearMaskedCoupling(nn.Module):
    """ Modified RealNVP Coupling Layers per the MAF paper """
    def __init__(self, input_size, hidden_size, n_hidden, mask, cond_label_size=None):
        super().__init__()

        self.register_buffer('mask', mask)

        # scale function
        s_net = [nn.Linear(input_size + (cond_label_size if cond_label_size is not None else 0), hidden_size)]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)

        # translation function
        self.t_net = copy.deepcopy(self.s_net)
        # replace Tanh with ReLU's per MAF paper
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear): self.t_net[i] = nn.ReLU()

    def forward(self, x, y=None):
        # apply mask
        mx = x * self.mask

        # run through model
        s = self.s_net(mx if y is None else torch.cat([y, mx], dim=1))
        t = self.t_net(mx if y is None else torch.cat([y, mx], dim=1))
        u = mx + (1 - self.mask) * (x - t) * torch.exp(-s)  # cf RealNVP eq 8 where u corresponds to x (here we're modeling u)

        log_abs_det_jacobian = - (1 - self.mask) * s  # log det du/dx; cf RealNVP 8 and 6; note, sum over input_size done at model log_prob

        return u, log_abs_det_jacobian

    def inverse(self, u, y=None):
        # apply mask
        mu = u * self.mask

        # run through model
        s = self.s_net(mu if y is None else torch.cat([y, mu], dim=1))
        t = self.t_net(mu if y is None else torch.cat([y, mu], dim=1))
        x = mu + (1 - self.mask) * (u * s.exp() + t)  # cf RealNVP eq 7

        log_abs_det_jacobian = (1 - self.mask) * s  # log det dx/du

        return x, log_abs_det_jacobian

class BatchNorm(nn.Module):
    """ RealNVP BatchNorm layer """
    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))

    def forward(self, x, cond_y=None):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0) # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        # compute log_abs_det_jacobian (cf RealNVP paper)
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)

        return y, log_abs_det_jacobian.expand_as(x)

    def inverse(self, y, cond_y=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, log_abs_det_jacobian.expand_as(x)

class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def forward(self, x, y):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, y):
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians

class RealNVP(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size=None, batch_norm=True):
        super().__init__()

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # construct model
        modules = []
        mask = torch.arange(input_size).float() % 2
        for i in range(n_blocks):
            modules += [LinearMaskedCoupling(input_size, hidden_size, n_hidden, mask, cond_label_size)]
            mask = 1 - mask
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        return self.net(x, y)

    def inverse(self, u, y=None):
        return self.net.inverse(u, y)

    def log_prob(self, x, y=None):
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)

def train(model, dataloader, optimizer):

    for i, data in enumerate(dataloader):
        model.train()

        # check if labeled dataset
        if len(data) == 1:
            x, y = data[0], None
        else:
            x, y = data
            y = y.cuda()#.to(args.device)
            y = y.unsqueeze(-1)
        x = x.view(x.shape[0], -1).cuda()#.to(args.device)

        loss = - model.log_prob(x, y if y is not None else None).mean(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def generate(model, dataset_lam, args, step=None, n_row=10):
    model.eval()

    # conditional model
    if args.cond_label_size:
        samples = []
        labels = torch.eye(args.cond_label_size).cuda()#.to(args.device)

        for i in range(args.cond_label_size):
            # sample model base distribution and run through inverse model to sample data space
            u = model.base_dist.sample((n_row, args.n_components)).squeeze()
            labels_i = labels[i].expand(n_row, -1)
            sample, _ = model.inverse(u, labels_i)
            log_probs = model.log_prob(sample, labels_i).sort(0)[1].flip(0)  # sort by log_prob; take argsort idxs; flip high to low
            samples.append(sample[log_probs])

        samples = torch.cat(samples, dim=0)

    # unconditional model
    else:
        u = model.base_dist.sample((n_row**2, args.n_components)).squeeze()
        samples, _ = model.inverse(u)
        log_probs = model.log_prob(samples).sort(0)[1].flip(0)  # sort by log_prob; take argsort idxs; flip high to low
        samples = samples[log_probs]

    # convert and save images
    samples = samples.view(samples.shape[0], *args.input_dims)
    samples = (torch.sigmoid(samples) - dataset_lam) / (1 - 2 * dataset_lam)
    filename = 'generated_samples' + (step != None)*'_epoch_{}'.format(step) + '.png'
    save_image(samples, os.path.join(args.output_dir, filename), nrow=n_row, normalize=True)

###
# Important Variables and Settings
###
V_intro_epoch = 40
batch_size = 32
seed = 666
# Data config
stride = 10
data_files_regex = './../../All_ML_Training_Data/210905_SMD_decaalanine/SMD/output/rand_orein_npy_traj/*npy'
# Configurations for Energy Calculation
data_dir = "./../../V_Calculations/Test-6_full_system/data/"
psf_file = "full_da-1.3.prmtop"  # This is a special psf file with improper connectivity deleted
parameter_file = "full_da-1.3.prmtop" # bond, angles, dihedrals, electrostatics, lj; no 1-4, impropers or external


###
# Initialize Dataset
###
psf_file = os.path.join(data_dir, psf_file)
parameter_file = os.path.join(data_dir, parameter_file)
traj_files = glob.glob(data_files_regex)
training_dataset = SystemDataset(psf_file, parameter_file, traj_files, colvar)
num_atoms = training_dataset.num_atoms
num_frames = training_dataset.num_frames

###
# Initialize RealNVP Model
###
n_blocks = 2
input_size = num_atoms*3
hidden_size = num_atoms
n_hidden = 3
model = RealNVP(n_blocks, input_size, hidden_size, n_hidden, cond_label_size=1).cuda()

###
# Begin Pre-Training
###
max_epochs_pre = 1
pretrain_dataset = SystemDataset(psf_file, parameter_file, traj_files, colvar, pretrain=True)
pretrain_dataloader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
print("Pre-Training Dataset size:", len(pretrain_dataset))
learning_rate = 1e-2
pretrain_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
pretrain_loss = []
pretrain_training_data = []
for epoch in range(max_epochs_pre):
    print(f"\n===> Starting pre-train epoch {epoch} <===")
    if epoch == V_intro_epoch + 1:
        pretrain_optimizer = optim.Adam(model.parameters(), lr=1e-4)
    start = time.time()
    train(model, pretrain_dataloader, pretrain_optimizer)
    end = time.time()
    torch.cuda.empty_cache()
    print(f"\n===> Finished Epoch {epoch} in {end-start:.2f} s <===\n")
    print(f"Pre-Training is {(1+epoch)/(max_epochs_pre)*100:.4f} % complete")

print('Done')
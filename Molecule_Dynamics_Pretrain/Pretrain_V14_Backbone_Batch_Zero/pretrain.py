##
# Pretrain V12.2: No Dropout
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
batch_size = 16

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
        if frame_num == 0:
            pretrain_dataset.append((frame_num, X[frame_num,:,:]))

new_dataset = []
for batch in range(int(len(dataset)/batch_size)):

    batched = []
    for item in dataset[batch*batch_size:batch*batch_size + batch_size]:

        fnum, data = item
        new_data = np.concatenate([[fnum], data.reshape(120)],-1).reshape(121)
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
        self.d1 = nn.Dropout(p=0.2)
        self.mlp2 = nn.Linear(50,75)
        self.d2 = nn.Dropout(p=0.4)
        self.mlp3 = nn.Linear(75,120)

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
        self.mlp1 = nn.Linear(250, 10)
        self.mlp2 = nn.Linear(10,1)

    def forward(self,frame, x):
        #x = torch.cat([frame, x],-1)
        x = frame
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
            #time = torch.tensor(time).float().cuda()
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

colvar = {
    "name": "E2End Harm",
    "fk": 1.0,
    "cent_0": 12.0,
    "cent_1": 34.0,
    "T": 50000/50,
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

class PotentialEnergyDataset(torch.utils.data.Dataset):

    def __init__(self, data, colvar, transform=None):

        self.data = data
        self.transform = transform
        self.sys_decal = Energy(data_dir, psf_file, parameter_file, colvar=colvar)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        frame_num, frame = sample

        frame = torch.tensor(frame).float().cuda()
        potential, phi = self.sys_decal.calc_energy(frame.view(40,3,1),frame_num)
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

pretrain_dataset = PotentialEnergyDataset(pretrain_dataset, colvar)

pretrain_dataloader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

# Begin Pre-Training
##
import torch.optim as optim

max_epochs = 30

max_steps = 1002
pretrain_loss = []
import time
print(len(pretrain_dataloader))

sys_decal = Energy(data_dir, psf_file, parameter_file, colvar=colvar)

min_val = 1e9
for q, learning_rate in enumerate([1e-2, 1e-2,1e-2, 1e-3, 1e-3, 1e-3,1e-3,1e-4]):
    pretrain_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    for epoch in range(max_epochs):

        epoch_pretrain_loss = []
        for i, data in enumerate(pretrain_dataloader):
            #start = time.time()
            data, t = data
            alpha = torch.tensor(epoch/max_epochs).float().cuda()
            one = torch.tensor(1.0).float().cuda()
            bonds_factor = torch.tensor(1.0).float().cuda()
            angle_factor = torch.tensor(1.0).float().cuda()
            dihedral_factor = torch.tensor(1.0).float().cuda()
            improper_factor = torch.tensor(1.0).float().cuda()
            lj_factor = torch.tensor(1.0).float().cuda()
            electrostatics_factor = torch.tensor(1.0).float().cuda()
            repulsion_factor = torch.tensor(1.0).float().cuda()
            dis_factor = torch.tensor(1.0).float().cuda()

            x = data['frame']
            bsize = x.size()[0]
            x = x.view(bsize, 40*3)
            target_phi = data['phi']
            z = torch.normal(0,1,size=(bsize,32)).cuda()
            t = t.unsqueeze(-1).float().cuda()/max_steps
            #z = torch.cat((t,z),1)

            pred_x = decoder(z)

            pretrain_optimizer.zero_grad()
            total_pot = torch.zeros(1,1).cuda()
            total_phi = torch.zeros(1,1).cuda()

            for px, t_phi, pt in zip(pred_x,target_phi, t):
                potential, phi = sys_decal.calc_energy(px.view(40,3,1),pt)
                total_phi +=  F.mse_loss(phi, t_phi)
                for key in potential[0].keys():
                    if key == 'bonds':
                        total_pot += bonds_factor*potential[0][key]/batch_size
                    if key == 'angles':
                        total_pot += angle_factor*potential[0][key]/batch_size
                    if key == 'dihedrals':
                        total_pot += dihedral_factor*potential[0][key]/batch_size
                    if key == 'impropers':
                        total_pot += improper_factor*potential[0][key]/batch_size
                    if q > 1:
                        if key == 'lj':
                            total_pot += lj_factor*potential[0][key]/batch_size
                        if key == 'electrostatics':
                            total_pot += electrostatics_factor*potential[0][key]/batch_size
                        if key == 'repulsion':
                            total_pot += repulsion_factor*potential[0][key]/batch_size
            dist_pred = torch.cdist(pred_x.view(bsize,40,3),pred_x.view(bsize,40,3))
            dist_target = torch.cdist(x.view(bsize,40,3),x.view(bsize,40,3))
            recon_loss = dis_factor*F.mse_loss(dist_pred, dist_target)
            #total_loss = alpha*total_pot + (one - alpha)*recon_loss
            total_loss = total_pot + 1e-3*recon_loss
            total_loss.backward()
            clipping_value = 1 # arbitrary value of your choosing
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), clipping_value)
            epoch_pretrain_loss.append(total_loss.item()) 
            pretrain_optimizer.step()
            #end = time.time()
            #print(str(end-start) +'s')
        
        l = np.mean(epoch_pretrain_loss)
        if l < min_val:
            min_val = l
            torch.save(decoder.state_dict(), 'pretrain-decoder.pt')
            print(l)
        pretrain_loss.append(l)

plt.figure()
plt.plot(range(len(pretrain_loss)), pretrain_loss)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.savefig('pretrain_loss.png')

decoder.load_state_dict(torch.load('pretrain-decoder.pt'))
decoder.eval()
# Go through the reaction coordinate of the trajectory
max_generation_steps = 20
predictions = []
for t in range(max_generation_steps):
    gen_frame = decoder.generation_step(t, max_generation_steps)
    predictions.append(gen_frame.view(40,3))

predictions = torch.stack(predictions)
predictions = predictions.cpu().detach().numpy()
# Save predictions into VMD format
frame_num = predictions.shape[0]

nAtoms = "40"
outName = "Pretrain.xyz"
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

print("=> Finished Pre-Train Generation <=")

#torch.save(decoder.state_dict(), 'pretrain-decoder.pt')
print('Pretrain Done')
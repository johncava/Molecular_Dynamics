# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski
# Modified by Nick

import torch, argparse
import numpy as np

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(""))
THIS_DIR = os.path.join(THIS_DIR, "Mol-HNN-cuda-v4.2")
# PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(PARENT_DIR)

from cuda_nn_models import *
from cuda_hnn import HNN
from cuda_utils import L2_loss, to_pickle, from_pickle


####### DATA PREPARATION STEP. NOT NECESSARY AFTER THE FIRST RUN

##
# Loading in the whitened Dataset
##
num_trajectories = 100
num_atoms = 40

import glob


files = glob.glob('./../../All_ML_Training_Data/210905_SMD_decaalanine/SMD/output/processed_orient/*.npy')
dataset = []
for num, file_ in enumerate(files):
    X_positions = np.load(file_)
    X = X_positions
    dataset.append(X)
    if len(dataset) == num_trajectories:
        break

dataset = np.array(dataset)
dataset = dataset.reshape(num_trajectories, -1, num_atoms*3)

### Preprocessing to create the training step

x_dataset = [] ## vector of position and momentum
dx_dataset = [] ## vector of change in position and momentum
## Getting the velocity of each trajectory
for traj_idx in range(num_trajectories):
    traj = dataset[traj_idx]
#     traj = dataset[0]
    num_datapoints = traj.shape[0] - 1

    momenta = []
    for i in range(traj.shape[0] - 1):
        dx = traj[i+1] - traj[i]
        momenta.append(dx)

    momenta = np.array(momenta) ## convert to an np array

    ## stack the coordinates so that we have all the positions and all the momentums
    coords = np.stack((traj[:-1], momenta), 1) ## shape: (19999, 2, 120)
    coords = coords.reshape(num_datapoints, 2 * 120) ## shape: (19999, 240)
    
    ## Now we need to calculate the change in coords (positions and momentums)
    delta_coords = []
    for frame in range(coords.shape[0] - 1):
        dx = coords[frame+1] - coords[frame]
        delta_coords.append(dx)
    delta_coords = np.array(delta_coords)
    
    ## appending to the dataset
    x_dataset.append(coords[:-1])
    dx_dataset.append(delta_coords)


x_dataset = np.array(x_dataset) ## shape of (100, 19998, 240)
dx_dataset = np.array(dx_dataset) ## shape of (100, 19998, 240)

x_dataset = x_dataset.reshape(-1, 240) ## shape of (1999800, 240)
dx_dataset = dx_dataset.reshape(-1, 240) ## shape of (1999800, 240)

'''
# x_dataset[0] + dx_dataset[0] == x_dataset[1] ## This is just a sanity check that we can predict the next step

## just for quick load don't need this

#np.save("whitened_x_dataset.npy", x_dataset)
#np.save("whitened_dx_dataset.npy", dx_dataset)


######## LOADING THE DATA

# x_dataset = np.load("whitened_x_dataset.npy")
# dx_dataset = np.load("whitened_dx_dataset.npy")

'''

x = torch.tensor(x_dataset, requires_grad=True, dtype=torch.float32).cuda()
print(x.size())
dxdt = torch.Tensor(dx_dataset).cuda()

batch_size = 100
epochs = 3
total_steps = int(x_dataset.shape[0] / 100 * epochs)
total_steps



#### LOADING THE PARAMETERS

# arrange data
def get_args():
    return {'input_dim': 240, # 40 atoms, each with q_x, q_y, p_z, p_y
         'hidden_dim': 200,
         'learn_rate': 1e-3,
         'input_noise': 0.1, ## NO INPUT NOISE YET
         'batch_size': 100,
         'nonlinearity': 'leaky',
         'total_steps': total_steps, ## 3 epochs effectively i guess
         'field_type': 'helmholtz', ## change this? solenoidal
         'print_every': 200,
         'verbose': True,
         'name': '2body',
         'baseline' : False,
         'seed': 0,
         'save_dir': '{}'.format(THIS_DIR),
         'fig_dir': './figures'}


class ObjectView(object):
    def __init__(self, d): self.__dict__ = d

args = ObjectView(get_args())


#### THIS IS WHERE YOU INTERCEPT IF YOU WANT TO USE A DIFFERENT MODEL
output_dim = 2
nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
num_particles = 40
channel_size = 3
hidden_size = 32
output_size = 2
#nn_model = GATModel(channel_size, hidden_size, output_size).cuda()
#nn_model = SchNet(num_particles).cuda()
model = HNN(args.input_dim, differentiable_model=nn_model,
        field_type=args.field_type, baseline=args.baseline)
optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=0)


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

potential_keys = [['bonds','angles','dihedrals'] for _ in range(args.total_steps+1)]
potential_factors = [(0.6,0.2,0.1)] + [(0.1,0.1,0.1) for _ in range(args.total_steps+1)]
### Training the data

# vanilla train loop
stats = {'train_loss': [], 'test_loss': []}
max_num = 1
import time
import random
for step in range(args.total_steps+1):

    #start = time.time()

    if step % 10 == 0:
        # train step
        ixs = torch.randperm(x.shape[0])[0]
        xi = x[ixs].unsqueeze(0)
        dxdt_hat = model.time_derivative(xi)
        dxdt_hat += args.input_noise * torch.randn(xi.shape).cuda() # add noise, maybe
        
        loss = L2_loss(dxdt[ixs], dxdt_hat)

        new_x = xi + dxdt_hat
        new_x = new_x[:,:120]
        # determine the potential energy
        p_factors = potential_factors[step]
        bonds_factor = torch.tensor(p_factors[0]).float().cuda()
        angle_factor = torch.tensor(p_factors[1]).float().cuda()
        dihedral_factor = torch.tensor(p_factors[2]).float().cuda()

        sys_decal = Energy(data_dir, psf_file, parameter_file)  
        potential = sys_decal.calc_energy(new_x.view(40,3,1))
        total_pot = torch.zeros(1,1).cuda()
        for key in potential_keys[step]:
            if key == 'bonds':
                total_pot += bonds_factor*potential[0][key]
            elif key == 'angles':
                total_pot += angle_factor*potential[0][key]
            elif key == 'dihedrals':
                total_pot += dihedral_factor*potential[0][key]
        total_loss = loss + total_pot 
        total_loss.backward()
        clipping_value = 1 # arbitrary value of your choosing
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)

        optim.step()
        optim.zero_grad()

        del sys_decal
        del new_x
        del potential
        del total_pot

        del bonds_factor
        del angle_factor
        del dihedral_factor
    else:
        # train step
        ixs = torch.randperm(x.shape[0])[:args.batch_size]
        dxdt_hat = model.time_derivative(x[ixs])
        dxdt_hat += args.input_noise * torch.randn(*x[ixs].shape).cuda() # add noise, maybe
        
        loss = L2_loss(dxdt[ixs], dxdt_hat)
        loss.backward() 
        optim.step()
        optim.zero_grad()

    #end = time.time()
print("=> Finished Training <=")

'''
## Saving model
PATH = "white-MOLHNNv1.pt"
torch.save(model.state_dict(), PATH)




################
## Loading the Model
############

PATH = "white-MOLHNNv1.pt"
model.load_state_dict(torch.load(PATH))
'''
model.eval()

### Autoregressive
initial_frame = x[0]
a = initial_frame
frames = []

for i in range(10000):
    dx_hat = model.time_derivative(a.reshape(-1, 240))
    a = a + dx_hat * 0.005
    a = a + torch.randn(a.shape).float().cuda() * 0.1 # add noise, maybe
    
    new_frame = a.cpu().detach().numpy()
    new_frame = new_frame[:, :120].reshape(40, 3) ## converts the frame of coordinates into XYZ
    frames.append(new_frame)

frames = np.array(frames)



# Save predictions into VMD format
predictions = frames
frame_num = predictions.shape[0]

nAtoms = "40"
outName = "HNNv4.xyz"
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
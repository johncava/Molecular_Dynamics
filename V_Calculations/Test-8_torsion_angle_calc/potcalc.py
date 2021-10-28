#!/usr/bin/env python
'''
############################################################## 
# Author:               John Vant 
# Email:              jvant@asu.edu 
# Affiliation:   ASU Biodesign Institute 
# Date Created:          210919
############################################################## 
# Usage: Standalone python script.  Edit paths in main to run.
############################################################## 
# Notes: The user is required to change file paths to a PSF 
# file and yaml parameter file in main at the bottom of this
# script if ran in standalone mode.
# 
############################################################## 
'''
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
        # Reshape array if needed
        if not coords.shape == (self.num_atoms, 3, 1):
            coords = np.reshape(coords, (self.num_atoms, 3, 1))
        # Set positions for system object
        self.system.set_positions(coords)
        # Evaluate current energy and forces. Forces are modified in-place
        forces = Forces(self.parameters, cutoff=9, rfa=True, switch_dist=7.5)
        # Calculate torsion anlges
        my_phis = calc_phi(forces, self.system.pos, self.system.box)
        # print(forces.par.dihedrals.shape)
        print("myphis\n", my_phis.shape)
        Epot = forces.compute(self.system.pos, self.system.box, self.system.forces, returnDetails=True)
        if not self.colvar_name == None:
            if time == None:
                print("No time provided.  Exiting calculation.")
                exit()
            time = torch.tensor(time)
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
            energies = Epot
        else:
            energies = Epot[0][self.etype]
            energies = [energies]
        return energies


# Functions
def calc_phi(forces_obj, spos, sbox):
    _, _, r12 = calculate_distances(
        spos, forces_obj.par.dihedrals[:, [0, 1]], sbox
    )
    _, _, r23 = calculate_distances(
        spos, forces_obj.par.dihedrals[:, [1, 2]], sbox
    )
    _, _, r34 = calculate_distances(
        spos, forces_obj.par.dihedrals[:, [2, 3]], sbox
    )
    phi_list = evaluate_torsion(
        r12, r23, r34, forces_obj.par.dihedral_params
    )
    return phi_list


def calculate_distances(atom_pos, atom_idx, box):
    atom_pos = atom_pos[0]
    direction_vec = wrap_dist(atom_pos[atom_idx[:, 0]] - atom_pos[atom_idx[:, 1]], box)
    dist = torch.norm(direction_vec, dim=1)
    direction_unitvec = direction_vec / dist.unsqueeze(1)
    return dist, direction_unitvec, direction_vec


def wrap_dist(dist, box):
    if box is None or torch.all(box == 0):
        wdist = dist
    else:
        wdist = dist - box.unsqueeze(0) * torch.round(dist / box.unsqueeze(0))
    return wdist

                    
def evaluate_torsion(r12, r23, r34, torsion_params, explicit_forces=True):
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


# Parameters
colvar = {
    "name": "E2End Harm",
    "fk": 1.0,
    "cent_0": 12.0,
    "cent_1": 34.0,
    "T": 500000/50,
    "group1": [3],
    "group2": [98]
}

if __name__ == "__main__":
    print("Running script as master")
    # Set file paths here
    data_dir = "./data"
    psf_file = "full_da-1.3.prmtop"  # prmtop file made using Charmm params with chamber in parmed
    parameter_file = "full_da-1.3.prmtop" # contains bonds, angles, dihedrals, electrostatics, lj; no 1-4, impropers or external

    # Make energy calculation object
    sys_decal = Energy(data_dir, psf_file, parameter_file, colvar=colvar, device=None)
    # Load data
    traj = np.load("data/raw-traj_rep-0.npy")

    
    fcount = 0    
    for frame in traj:
        print(sys_decal.calc_energy(frame, fcount))
        fcount+=1

    print(sys_decal)
    print("all done! :)")

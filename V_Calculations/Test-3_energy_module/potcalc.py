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
        if not coords.shape == (self.num_atoms, 3, 1):
            coords = np.reshape(coords, (self.num_atoms, 3, 1))
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



if __name__ == "__main__":
    print("Running script as master")
    # Set file paths here
    data_dir = "./data"
    psf_file = "backbone-no-improp.psf"  # This is a special psf file with improper connectivity deleted
    parameter_file = "param_bb-3.0.yaml" # bond, angles, dihedrals, electrostatics, lj; no 1-4, impropers or external

    # Make energy calculation object
    sys_decal = Energy(data_dir, psf_file, parameter_file, device=None)
    # Load data
    traj = np.load("data/rand-orient-rep0-0.npy")
    for frame in traj:
        print(sys_decal.calc_energy(frame))

    print(sys_decal)
    print("all done! :)")

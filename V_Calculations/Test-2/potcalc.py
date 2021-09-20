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
# Notes: 
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

    def __init__(self, etype='all'):
        self.etype = etype

    def __str__(self):
        return f"Energy of type {self.etype} is: {self.calc_energy()} {self.UNITS}"

    def calc_energy(self):
        if self.etype == 'all':
            print("all")
        return energies



if __name__ == "__main__":
    print("hello I am not set up yet :(...")
    testdir = "./data/"
    precision = torch.float
    device = "cuda:0"
    psf_file = "backbone-no-improp.psf"
    pdb_file = "backbone.pdb"
    # xsc_file = "input.xsc"
    
    # Make Molecule object
    mol = Molecule(os.path.join(testdir, psf_file))  # Reading the system topology
    # mol.read(os.path.join(testdir, pdb_file))  # Reading the initial simulation coordinates
    #mol.read(os.path.join(testdir, xsc_file))  # Reading the box dimensions


    # Create Force Field object
    ff = ForceField.create(mol, os.path.join(testdir, "param_bb-3.0.yaml"))
    # My Nvidia driver was too old thus I disabled using the gpu
    #parameters = Parameters(ff, mol, precision=precision, device=device)
    parameters = Parameters(ff, mol, precision=precision)

    # Convert Moleculekit Molecule object to torchmd system object
    system = System(mol.numAtoms, nreplicas=1, precision=precision, device=None)
    # system.set_positions(mol.coords)

    # # Evaluate current energy and forces. Forces are modified in-place    
    # forces = Forces(parameters, cutoff=9, rfa=True, switch_dist=7.5)
    # Epot = forces.compute(system.pos, system.box, system.forces, returnDetails=True)
    # print(system.box)
    # print(type(mol.coords))

    # print(Epot)

    # Update positions
    ## reshape array
    oneframe = np.load("data/rand-orient-rep0-0.npy")[100]
    new_coord = np.reshape(oneframe, (len(oneframe),3,1))
    # print(new_coord)
    system.set_positions(new_coord)
    # Evaluate current energy and forces. Forces are modified in-place    
    forces = Forces(parameters, cutoff=9, rfa=True, switch_dist=7.5)
    Epot = forces.compute(system.pos, system.box, system.forces, returnDetails=True)
    print(Epot)

    # print(system.forces)

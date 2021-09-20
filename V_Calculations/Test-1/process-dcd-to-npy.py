#!/usr/bin/env python
import MDAnalysis
import numpy
import sys
import os.path

PDB = "../../All_ML_Training_Data/210905_SMD_decaalanine/SMD/output/0/my.pdb"
DCD = "../../All_ML_Training_Data/210905_SMD_decaalanine/SMD/output/0/smd_out.dcd"
OUT = "./data/out-1.npy"
SELECTION = "name CA"

out_file_format = os.path.splitext(OUT)[1] # Should be dcd or npy...

u = MDAnalysis.Universe(PDB, DCD)

system = u.select_atoms(SELECTION)

# Write a pdb of the system that can be used as a topology file to open the
# resulting trajectory in VMD for example.
system.write('%s.pdb'%os.path.splitext(OUT)[0])


out_array = []
for ts in u.trajectory:
    out_array.append(system.positions)
out_array = numpy.asarray(out_array)
print('Output array shape: (%d, %d, %d)'%out_array.shape)
numpy.save(OUT, out_array)


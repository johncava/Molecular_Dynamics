#!/usr/bin/env python
'''
############################################################## 
# Author:               John Vant 
# Email:              jvant@asu.edu 
# Affiliation:   ASU Biodesign Institute 
# Date Created:          211008
############################################################## 
# Usage: 
############################################################## 
# Notes: 
############################################################## 
'''
from MDAnalysis import Universe
import MDAnalysis.analysis.encore as encore
import numpy as np

replica=0
outdir = "raw_aligned_backbone_npy_traj"
PSF = "../../Build/da.psf"
num_replica = 50
# selection = "not (name NT HT1 HT2 HT3 and resid 10)"
selection = "backbone"

for replica in range(num_replica):
    DCD = "%d/smd_aligned.dcd"%replica
    sys = Universe(PSF, DCD)
    # print(dir(sys.select_atoms(selection)))
    # print(sys.select_atoms(selection).positions)
    all_atoms = sys.select_atoms(selection)
    mylist = []
    count = 0
    # print(sys.trajectory)
    for ts in sys.trajectory:
        # print(all_atoms.positions)
        mylist.append(all_atoms.positions)
        # if count == 10:
        #     break
        # else:
        #     count+=1
    print("DCD: %s\n"%DCD, np.array(mylist).shape)
    np.save("%s/raw-traj_rep-%d.npy"%(outdir,replica), np.array(mylist))

exit()

#!/usr/bin/env python

import numpy as np
from getPhiPsiDist import *
from getChunk import getChunk
from sys import argv

atomsPerRes = 4.

#wround = 0
wround = int(argv[1])
chunk = getChunk(wround)

fName = "Generation_round" + str(wround)  + "_15_32_5_" + str(chunk) + ".npy"
coords = np.load(fName)

nFrames, nAtoms, dontneed = coords.shape
nRes = nAtoms/atomsPerRes

phi_new = np.zeros((int(nFrames), int(nRes)))
psi_new = np.zeros((int(nFrames), int(nRes)))

for frameNum in range(nFrames):
    frame = coords[frameNum]
    phi_new[frameNum] = getPhiVals(frame)
    psi_new[frameNum] = getPsiVals(frame)

phi_new[:,0] = 0
psi_new[:,(int(nRes)-1)] = 0

np.savetxt("Phi_" + str(wround) + ".dat", phi_new)
np.savetxt("Psi_" + str(wround) + ".dat", psi_new)





import numpy as np
import pandas as pd
from getCoordsFromXYZ_new import getCoordsFromXYZ
from readPDB_and_PSF import *
from getChunk import getChunk
from sys import argv

def convToPDB(cood):
    objShape = cood.shape
    if len(objShape) > 2:
        print("Object has multiple frames")
    
    nAtoms = objShape[0]
    # Assume backbone
    nRes = int(nAtoms/4)
    #print("Number of residues: ", nRes)
    atomNameBase = pd.Series(['N', 'CA', 'C', 'O'])
    atomName = pd.Series(np.tile(atomNameBase, nRes))
    resNameBase = 'ALA'
    resName = pd.Series(np.tile(resNameBase, nAtoms))
    chainBase = 'C'
    chain = pd.Series(np.tile(chainBase, nAtoms))
    residBase = np.arange(1, nRes+1)
    resid = pd.Series(np.tile(residBase, 4))
    occu = np.ones(nAtoms)
    beta = np.zeros(nAtoms)
    df = pd.DataFrame({"AtomName": atomName, "Resname": resName, "Chain": chain, "Resid": resid, "X": cood[:,0], "Y": cood[:,1], "Z": cood[:,2], "Occu": occu, "Beta": beta})
    return df

wround = int(argv[1])
chunk = getChunk(wround)
fName = 'Generation_round' + str(wround) + '_15_32_5_' + str(chunk) + '.xyz'
Coordinates, nAtoms = getCoordsFromXYZ(fName)
cood = np.array(Coordinates)

for frameNum in np.arange(cood.shape[0]):
    frame = convToPDB(cood[frameNum])
    writePDB(frame, "PDBs/frame_" + str(frameNum) + ".pdb")


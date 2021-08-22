import numpy as np
from scale_features import *
from getChunk import *
from getBucket import *

def calcDist(Coordinates_temp, allTraj=False):
    nFrames = Coordinates_temp.shape[0]
    nAtoms = Coordinates_temp.shape[1]
    allDist = np.zeros((nFrames,nAtoms))
    for frame in np.arange(nFrames):
        thisFrame = Coordinates_temp[frame]
        distList = np.linalg.norm(np.diff(thisFrame, axis=0), axis=1)
        last = distList[-1]
        distList2 = np.append(distList, last)
        allDist[frame] = distList2
    allDist2 = allDist.reshape(nFrames, nAtoms, 1)
    if allTraj:
        minDist = np.min(allDist)
        maxDist = np.max(allDist)
    else:
        t = np.loadtxt("min_max_dist.txt")
        minDist = t[0]
        maxDist = t[1]
        if np.min(allDist2) < minDist:
            print("min too low")
        if np.max(allDist2) > maxDist:
            print("max too high")
        t = np.insert(allDist2, 0, minDist, axis=1)
        allDist2 = np.insert(t, 0, maxDist, axis=1)
        nAtoms = nAtoms + 2
    # Now scale the distances. Again, same as before
    nd_temp2 = normalizedData(allDist2)
    nd_temp2.scaleIt(numDims=1)
    dist_temp = nd_temp2.scaled_dat
    dist_temp2 = dist_temp.reshape(nFrames, nAtoms)
    if allTraj:
        pass
    else:
        test = np.delete(dist_temp2, 0, axis=1)
        dist_temp2 = np.delete(test, 0, axis=1)
    return dist_temp2, nd_temp2, minDist, maxDist

def procPhiPsi(phi, ang):
    if ang == 'phi':
        phi[:,0] = phi[:,1]
    else:
        phi[:,-1] = phi[:,-2]
    phi = phi/180
    nFrames = phi.shape[0]
    nAtoms = phi.shape[1]
    phi_Dat = np.repeat(phi, 4).reshape(nFrames,4*nAtoms)
    return phi_Dat



def readSeed_save(wround):
    chunk = getChunk(wround)
    trunc_start, trunc_stop = getBucket(chunk)
    # Again, make sure the path is ok.
    # This time, take a trajectory that was NOT used in training
    # Rest is the same as readData.py used in training, EXCEPT the last line
    fName = "./../../10_deca_alanine_main_dataset/10_deca_alanine/99/backbone.npy"

    rawCood_temp = np.load(fName)
    rawCood_temp2 = rawCood_temp[trunc_start:trunc_stop]

    nFrames = rawCood_temp2.shape[0]
    nAtoms = rawCood_temp2.shape[1]
    dist_temp2, nd_temp2, minDist, maxDist = calcDist(rawCood_temp2, allTraj=True)

    nd_temp = normalizedData(rawCood_temp2)
    nd_temp.scaleIt()
    Coordinates_temp = nd_temp.scaled_dat
    Coordinates_temp2 = np.dstack((Coordinates_temp, dist_temp2))

    # Again, make sure the path is ok
    PhiPsi = np.load("./../../10_deca_alanine_main_dataset/10_deca_alanine/99/allPhiPsi.npy")[::10]
    PhiPsi2 = PhiPsi[trunc_start:trunc_stop]
    phi_temp = PhiPsi2[:,:,0]
    psi_temp = PhiPsi2[:,:,1]
    phi = phi_temp/180
    psi = psi_temp/180

    rawCood = np.dstack((Coordinates_temp2, phi, psi))

    np.save("seed.npy", rawCood)
    minMax = np.row_stack((minDist, maxDist))
    np.savetxt("min_max_dist.txt", minMax)
    print("Seed saved for bucket: ", str(chunk))
    print("Seeding from frame ", trunc_start, " to frame ", trunc_stop)
    return nd_temp, nd_temp2

def readSeed_round0():
    dat = np.load("seed.npy")
    return dat[:15]

def readSeed_round1(wround, chunk):
    # Again, make sure the path is ok.
    # Make sure to use the same one as in round0
    rawCood = np.load("seed.npy")
    rawCood_temp = rawCood[:,:,:3]

    # Take first 20 frames of original seed
    Coordinates_0 = rawCood[:20]
    Coordinates_0_temp = rawCood_temp[:20]

    # Then take the next 10 frames generated in round 0
    Coordinates_1_temp = np.load("Generation_round" + str(wround - 1) + "_unscaled.npy")
    Coordinates_1_real = np.load("Generation_round" + str(wround - 1) + "_15_32_5_" + str(chunk) + ".npy")

    # Merge these to get a 30-frame data, where first 20 frames are seed, next 10 frames are generated
    Coordinates_temp = np.row_stack((Coordinates_0_temp, Coordinates_1_temp))

    dist_temp2, nd_temp2, dontNeed, dontNeed2 = calcDist(Coordinates_1_real, allTraj=False)
    Coordinates_1_4D = np.dstack((Coordinates_1_temp, dist_temp2))

    # Now, read phi of the 10 frames generated in round 0
    phi_1 = np.loadtxt("Phi_0.dat")
    phiDat = procPhiPsi(phi_1, ang='phi')

    # Now, read psi of the 10 frames generated in round 0
    psi_1 = np.loadtxt("Psi_0.dat")
    psiDat = procPhiPsi(psi_1, ang='psi')

    # Combine phi and psi
    phipsi_1_temp = np.dstack((phiDat, psiDat))
    #phipsi_temp = np.row_stack((phipsi_0_temp, phipsi_1_temp))

    Generated = np.dstack((Coordinates_1_4D, phipsi_1_temp))

    # Create the 6D dataset
    # And NOW comes the main part. We use frames 10 - 25 of the 30-frame data
    Coordinates = np.row_stack((Coordinates_0, Generated))[10:25]

    # This has to change later
    nd_temp, dontneed = readSeed_save(wround)
    return Coordinates, nd_temp, nd_temp2

def read_round2_noseed(wround, chunk_2, chunk_1):
    # Read the 10 frames generated two rounds ago
    Coordinates_1 = np.load("Generation_round" + str(wround-2) + "_unscaled.npy")
    # Read the 10 frames generated in the previous round
    Coordinates_2 = np.load("Generation_round" + str(wround-1) + "_unscaled.npy")
    # Combine and take the first 15
    Coordinates_temp = np.row_stack((Coordinates_1, Coordinates_2))[:15]

    dist_temp2, nd_temp2, dontNeed, dontNeed2 = calcDist(Coordinates_temp, allTraj=False)

    # Read phi generated two rounds ago
    phi_1 = np.loadtxt("Phi_" + str(wround-2) + ".dat")
    phi_1_Dat = procPhiPsi(phi_1, ang='phi')

    # Read phi generated two rounds ago
    phi_2 = np.loadtxt("Phi_" + str(wround-1) + ".dat")
    phi_2_Dat = procPhiPsi(phi_2, ang='phi')

    # Combine the phi's, and take the first 15
    phi = np.row_stack((phi_1_Dat, phi_2_Dat))[:15]

    # Read psi generated two rounds ago
    psi_1 = np.loadtxt("Psi_" + str(wround-2) + ".dat")
    psi_1_Dat = procPhiPsi(psi_1, ang='psi')

    # Read psi generated in the previous round
    psi_2 = np.loadtxt("Psi_" + str(wround-1) + ".dat")
    psi_2_Dat = procPhiPsi(psi_2, ang='psi')

    # Combine the psi's and take the first 15
    psi = np.row_stack((psi_1_Dat, psi_2_Dat))[:15]

    # Create the 6D data
    Coordinates = np.dstack((Coordinates_temp, dist_temp2, phi, psi))

    # This has to change later
    nd_temp, dontneed = readSeed_save(wround)
    return Coordinates, nd_temp, nd_temp2


def read_round2_newseed(wround, lead_time):
    # Seeding event
    ground_truth = np.load("seed.npy")

    #start = wround * 10
    #stop = start + lead_time
    print("New bucket in wround ", wround)
    #print("Seeding from frame ", start, " to frame ", stop)

    Coordinates = ground_truth[:15]

    # This has to change later
    nd_temp, nd_temp2 = readSeed_save(wround)
    return Coordinates, nd_temp, nd_temp2


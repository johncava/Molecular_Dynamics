import numpy as np
from scale_features import *

def getData(num):
    NUM = str(num).zfill(2)

    fName = "./../../10_deca_alanine_main_dataset/10_deca_alanine/" + NUM + "/backbone.npy"
    rawCood_temp = np.load(fName)
    nFrames = rawCood_temp.shape[0]
    nAtoms = rawCood_temp.shape[1]
    allDist = np.zeros((nFrames,nAtoms))
    for frame in np.arange(nFrames):
        thisFrame = rawCood_temp[frame]
        distList = np.linalg.norm(np.diff(thisFrame, axis=0), axis=1)
        last = distList[-1]
        distList2 = np.append(distList, last)
        allDist[frame] = distList2
    Coordinates_temp2 = np.dstack((rawCood_temp, allDist))

    PhiPsi = np.load("./../../10_deca_alanine_main_dataset/10_deca_alanine/" + NUM + "/allPhiPsi.npy")[::10]
    phi_temp = PhiPsi[:,:,0]
    psi_temp = PhiPsi[:,:,1]
    phi = phi_temp/180
    psi = psi_temp/180
    rawCood2 = np.dstack((Coordinates_temp2, phi, psi))
    return rawCood2

def readData(l, trainList, trunc_start, trunc_stop, noBucket=False):
    ensemble_size = len(trainList)
    print("Ensemble size sent to readData is: ", ensemble_size)
    for num in trainList:
        allDat_notScaled = getData(num)
        if noBucket:
            pass
        else:
            allDat_notScaled_bucket = allDat_notScaled[trunc_start:trunc_stop]

        nFrames, nAtoms, six = allDat_notScaled_bucket.shape

        XYZ = allDat_notScaled_bucket[:,:,:3]
        nd_temp = normalizedData(XYZ)
        nd_temp.scaleIt()
        XYZ_scaled = nd_temp.scaled_dat
 
        dist = allDat_notScaled_bucket[:,:,3]
        print("dist shape: ", dist.shape)
        dist_unscaled = dist.reshape(nFrames, nAtoms, 1)
        print("dist_unscaled shape: ", dist_unscaled.shape)
        nd_temp2 = normalizedData(dist_unscaled)
        nd_temp2.scaleIt(numDims=1)
        dist_temp = nd_temp2.scaled_dat
        dist_scaled = dist_temp.reshape(nFrames, nAtoms)

        phipsi = allDat_notScaled_bucket[:,:,4:]
        allDat_scaled = np.dstack((XYZ_scaled, dist_scaled, phipsi))
        l.append(allDat_scaled)

    return l, nd_temp, nd_temp2




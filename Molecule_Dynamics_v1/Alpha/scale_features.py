#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Not much to comment here

# normalizedData is for scaling a given numpy array.
# inversedData is for rescaling an object of the normalizedData class

# Please let me know if you have questions

class normalizedData(object):
    def __init__(self, raw_dat):
        self.dat = raw_dat
        self.nFrames = raw_dat.shape[0]
        self.nAtoms = raw_dat.shape[1]
        self.X = 0
        self.Y = 1
        self.Z = 2
        
    def normalize(self,arr):
        t2 = arr.reshape(arr.shape[0], 1)
        # train the normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(t2)
        # normalize the dataset and print
        normalized = scaler.transform(t2)
        #print(normalized)
        return normalized, scaler
    
    def scaleIt(self, numDims=3):
        self.scaled_dat = np.zeros((self.nFrames, self.nAtoms, numDims))
        self.scalerList = []
        dimList = [self.X, self.Y, self.Z]
        dimList = dimList[:numDims]
        for atom in range(self.nAtoms):
            self.scalerList.append([])
            for coord in dimList:
                self.scalerList[atom].append([])
                t = self.dat[:,atom,coord]
                p, p_scaler = self.normalize(t)
                self.scaled_dat[:,atom,coord] = p[:,0]
                self.scalerList[atom][coord] = p_scaler


class inversedData(normalizedData):
    def __init__(self, normalizedData):
        try:
            self.scalerList = normalizedData.scalerList
        except:
            print("This instance of normalized data is not scaled yet")
    
    def inverseIt(self, scaled_dat):
        X = 0
        Y = 1
        Z = 2
        self.nFrames = scaled_dat.shape[0]
        self.nAtoms = scaled_dat.shape[1]
        self.inversed_dat = np.zeros((self.nFrames, self.nAtoms, 3))
        for atom in range(self.nAtoms):
            for coord in [X, Y, Z]:
                #print("Scaled dat shape: ", scaled_dat.shape)
                p = scaled_dat[:, atom, coord]
                p2 = p.reshape(p.shape[0], 1)
                inversed = self.scalerList[atom][coord].inverse_transform(p2)
                self.inversed_dat[:,atom,coord] = inversed[:,0]



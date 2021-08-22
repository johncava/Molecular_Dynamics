import sys
import shutil
import os
import random
# from readData import *
from getBucket import getBucket
from sys import argv
from numpy import mean, sqrt, square, arange
import time

import numpy as np
import torch as pt
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F

from readSeed import *
from getChunk import getChunk
from getBucket import *
from scale_features import *
from getPhiPsiDist import *

sys.argv = ['',0]

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"
    
######################################################################


### 
# These are the parameters to change
###
chunk = 4 ## determines which seeding frames, if chunk=4 then seeding starts at frames 2900 (basically 3000)
PATH = "pt_trained_models/v1.pt" ## where the original model was saved
outName = "pt_generated/v1l01.xyz" ## where you want to write the xyz file


# This is the same as before, except that now there is no ensemble_size
# We only generate one trajectory at a time
class ToySequenceData(object):

    def __init__(self, raw_data, max_seq_len, lead_time) :

        self.data = []
        self.labels = []
        self.seqlen = []
        self.y_previous=[]
        self.lead_time = lead_time

        for i in range(max_seq_len, len(raw_data)):
            for j in range(len(raw_data[i])):
                # This loop runs nAtoms times 
                s=[]
                self.seqlen.append(max_seq_len)
                for k in range(i-max_seq_len, i):
                    s.append([])
                    for l in range(len(raw_data[k][j])):
                        s[max_seq_len-(i-k)].append(raw_data[k][j][l])
                self.data.append(s)

                # Note that our y is only 3-dimensional
                symbols_out_onehot=[]
                for k in range(3):
                    symbols_out_onehot.append(raw_data[i][j][k])
                self.labels.append(symbols_out_onehot)

                # Note that our y is only 3-dimensional
                symbols_out_onehot=[]
                for k in range(3):
                    symbols_out_onehot.append(raw_data[i-1][j][k])
                self.y_previous.append(symbols_out_onehot)



        self.batch_id = 0

    def next(self, batch_size):
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])

        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))

        return batch_data, batch_labels, batch_seqlen,self.batch_id


    def getAll(self):                                                            
        return self.data, self.labels, self.y_previous,self.seqlen,self.batch_id
    

    
# Device configuration
device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size1, LSTM_numlayers, output_size):
        super(RNN, self).__init__()
        
        self.hsize1 = hidden_size1
        self.num_layers = LSTM_numlayers
        self.lstm1 = nn.LSTM(input_size, hidden_size1, LSTM_numlayers, batch_first = True)
        self.fc_out = nn.Linear(hidden_size1, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        h01, c01 = self.init_lstm_hidden(self.num_layers,  x.shape[0], self.hsize1)
        x, _ = self.lstm1(x, (h01, c01))
        
        last_hidden_state = x[:, -1, :] ## getting the last hidden state
        out = self.fc_out(last_hidden_state)
        out = pt.sigmoid(out)
        
        mu = out[: , :3]
        sigma = out[: , 3:]
        return mu, sigma
    
    
    def init_lstm_hidden(self, numlayers, batch_size, hidden_size):
        h0 = pt.randn(numlayers , batch_size , hidden_size).double().to(device)
        c0 = pt.randn(numlayers , batch_size , hidden_size).double().to(device)
        return h0, c0

## input size of a single sequence object
input_size = 6
## intermediate size of the LSTM
hidden_size1 = 32
LSTM_numlayers = 1
output_size = 6 ## output size of a single object


model = RNN(input_size, hidden_size1, LSTM_numlayers, output_size).to(device)
model.load_state_dict(pt.load(PATH))
model.eval()



#### 
# LOADING IN SOME NECESSARY FUNCTIONS
####

# Some necessary functions

def calc_phi_psi(predicted_frames):
    coords = predicted_frames

    atomsPerRes = 4.
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
    return phi_new, psi_new
    
    
    
def Loop_readSeed1(wround, chunk, real_pred_frames, scaled_pred_frames):
    rawCood = np.load("seed.npy")
    rawCood_temp = rawCood[:,:,:3]

    real_pred_frames = real_pred_frames[-10:] # last 10 frames in unscaled_space
    scaled_pred_frames = scaled_pred_frames[-10:] # last 10 frames in scaled space

    # Take first 20 frames of original seed
    Coordinates_0 = rawCood[:20]

    ## calculating distance from real frames
    dist_temp2, nd_temp2, dontNeed, dontNeed2 = calcDist(real_pred_frames, allTraj=False)
    
    print(dist_temp2.shape)
    ## adding distance
    Coordinates_1_4D = np.dstack((scaled_pred_frames, dist_temp2))

    ## caclulating phipsi from real frames
    phi_1, psi_1 = calc_phi_psi(real_pred_frames)
    phiDat = procPhiPsi(phi_1, ang='phi')
    psiDat = procPhiPsi(psi_1, ang='psi')

    phipsi_1_temp = np.dstack((phiDat, psiDat))

    ## adding phipsi
    Generated = np.dstack((Coordinates_1_4D, phipsi_1_temp))

    # Create the 6D dataset
    # And NOW comes the main part. We use frames 10 - 25 of the 30-frame data
    Coordinates = np.row_stack((Coordinates_0, Generated))[10:25]
    
    # This has to change later
#     nd_temp, dontneed = readSeed_save(wround)
    return Coordinates

    
def NLL_round2_noseed(wround, unscaled_pred_frames, scaled_pred_frames):
    
    ## the most recent 15 frames
    unscaled_Coordinates_temp = unscaled_pred_frames[-15:]
    scaled_Coordinates_temp = scaled_pred_frames[-15:]
    dist_temp2, nd_temp2, dontNeed, dontNeed2 = calcDist(unscaled_Coordinates_temp, allTraj=False)

    ## caclulating phipsi
    phi_1, psi_1 = calc_phi_psi(unscaled_Coordinates_temp)
    phiDat = procPhiPsi(phi_1, ang='phi')
    psiDat = procPhiPsi(psi_1, ang='psi')
    phipsi_1_temp = np.dstack((phiDat, psiDat))

    Coordinates = np.dstack((scaled_Coordinates_temp, dist_temp2, phipsi_1_temp))

    # This has to change later
    #nd_temp, dontneed = readSeed_save(wround)
    return Coordinates


def Loop_round2_newseed(lead_time):
    # Seeding event
    ground_truth = np.load("seed.npy")

    #start = wround * 10
    #stop = start + lead_time
#     print("New bucket in wround ", wround)
    #print("Seeding from frame ", start, " to frame ", stop)

    Coordinates = ground_truth[:15]

    # This has to change later
#     nd_temp, nd_temp2 = readSeed_save(wround)
    return Coordinates
    

    
####
# The actual generation part
###


lead_time = 15
maxSeqlength = 5

# In each round, we predict 10 frames. This is related to maxSeqlength and lead_time.
# I have forgotten the exact relationship, but something simple.
# Maybe (lead_time - maxSeqlength) is the number of frames we can predict in each round
nframes_pred = 10

# fName = "../11_forcedcd/00/backbone.npy"
# rawCood_temp = np.load(fName)
predicted_frames_real_space = np.zeros((1,40,3)) ## just to allow concatenation
scaled_pred_frames = np.zeros((1,40,3))


print("Seeding NEW CHUNK")
nd_temp, nd_temp2 = readSeed_save(wround = chunk * 97)
Coordinates = Loop_round2_newseed(lead_time)



model.double()
nframes_pred = 10
lambd = 1.0
predicted_frames_real_space = np.zeros((1,40,3)) ## just to allow concatenation
scaled_pred_frames = np.zeros((1,40,3))

wround = 0
while wround < 97:
    training_data = ToySequenceData(Coordinates, maxSeqlength, lead_time)
    all_x, all_y, all_y_previous, all_seqlen, batch_id = training_data.getAll()
    all_x = np.array(all_x)

    batch_x = all_x.reshape(-1, maxSeqlength, input_size) ## safety precaution
    torch_batch_x = pt.from_numpy(batch_x).double().to(device)
    mu, sigma = model(torch_batch_x)
    
    mu_np = mu.cpu().detach().numpy() 
    std_np = sigma.cpu().detach().numpy() 

    mux = mu_np[:, 0]
    muy = mu_np[:, 1]
    muz = mu_np[:, 2]
    stdx = std_np[:, 0]
    stdy = std_np[:, 1]
    stdz = std_np[:, 2]

    pred_x_all = mux + np.random.normal(0,1, len(stdx)) * stdx * lambd
    pred_y_all = mux + np.random.normal(0,1, len(stdy)) * stdy * lambd
    pred_z_all = mux + np.random.normal(0,1, len(stdz)) * stdz * lambd

    my_predictions_raw = np.dstack((pred_x_all, pred_y_all, pred_z_all))
    
    temp = my_predictions_raw.reshape(10, 40, 3)
    # This is where we invoke the inversedData class
    new = inversedData(nd_temp)
    new.inverseIt(temp)
    new_frames_real_space = new.inversed_dat
    
    
    predicted_frames_real_space = np.concatenate((predicted_frames_real_space, new_frames_real_space))
    scaled_pred_frames = np.concatenate((scaled_pred_frames, temp))
    
    if(wround == 0):
        print("Seeding 1")
        ### DO NOT FORGET THIS ONLY TAKES THE FIRST 10 FRAMES
        Coordinates = Loop_readSeed1(wround, chunk, predicted_frames_real_space, scaled_pred_frames)
    else:
        print("Seeding 2 noseed")
#                 Coordinates, nd_temp, nd_temp2 = NLL_round2_noseed(wround, predicted_frames_real_space, scaled_pred_frames)
        Coordinates= NLL_round2_noseed(wround, predicted_frames_real_space, scaled_pred_frames)

    wround += 1
    
###
# writing the xyz
###


## I remove the first frame cuz its some dummy placeholder i put
my_predictions = predicted_frames_real_space[1:]
frame_num = my_predictions.shape[0]

nAtoms = "40"
with open(outName, "w") as outputfile:
    for frame_idx in range(frame_num):
        
        frame = my_predictions[frame_idx]
        outputfile.write(str(nAtoms) + "\n")
        outputfile.write("generated by JK\n")

        atomType = "CA"
        for i in range(40):
            line = str(frame[i][0]) + " " + str(frame[i][1]) + " " + str(frame[i][2]) + " "
            line += "\n"
            outputfile.write("  " + atomType + "\t" + line)

print("=> Finished Generation <=")

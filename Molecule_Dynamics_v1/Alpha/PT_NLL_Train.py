import sys
import shutil
import os
import random
from readData import *
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


###
# Recording Time
###
start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


###
# Loading in the Data
### 
# This defines our creating of (x,y) pairs of training.
# Here, x is a time-series of coordinates. Size 
# of the time-series is decided by "maxSeqlength", which
# we refer to as "history". 
# y is the value of that coordinate in future. How much
# into the future we go is decided by "lead_time".

class ToySequenceData(object):
    
    def __init__(self, raw_data_all, max_seq_len, lead_time):
        
        self.data = []
        self.labels = []
        self.seqlen = []
        self.y_previous=[]
        self.lead_time = lead_time

        ensemble_size = len(raw_data_all)
        for ts in range(ensemble_size):
            raw_data = raw_data_all[ts]
            for i in range(max_seq_len, len(raw_data) - self.lead_time):
                for j in range(len(raw_data[i])):
                    s=[]
                    self.seqlen.append(max_seq_len)
                    for k in range(i-max_seq_len, i):
                        s.append([])

                        for l in range(len(raw_data[k][j])):
                            s[max_seq_len-(i-k)].append(raw_data[k][j][l])
                    self.data.append(s)


                    symbols_out_onehot=[]
                    #for k in range(len(raw_data[i][j])):
                    for k in range(3):
                        symbols_out_onehot.append(raw_data[i+self.lead_time][j][k])
                    self.labels.append(symbols_out_onehot)


                    symbols_out_onehot=[]
                    #for k in range(len(raw_data[i-1][j])):
                    for k in range(3):
                        symbols_out_onehot.append(raw_data[i-1][j][k])
                    self.y_previous.append(symbols_out_onehot)
                
        self.batch_id = 0


                                              
    def next(self, batch_size):                                                       
        if self.batch_id + batch_size >= len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
                                              
        return np.array(batch_data), np.array(batch_labels), batch_seqlen, self.batch_id
    
    
    def shuffle(self, seed = 0):
        print("Shuffling The Data")
        self.data, self.labels, self.y_previous = shuffle(self.data, self.labels, self.y_previous,random_state=seed)


    def getAll(self):                                                                                        
        return self.data, self.labels, self.y_previous,self.seqlen,self.batch_id
##############

#### THE PARAMETERS TO CHANGE
chunk = float(4) ## a chunk is every 1000 frames, so chunk 4 is roughly frames 3000~4000
train_model_PATH = "./pt_trained_models/v1.pt"
ensemble_size = 10 ## number of trajectories to train on, this is tuned by you



trunc_start, trunc_stop = getBucket(chunk)
print("Using frames: ", trunc_start, " to: ", trunc_stop)

allList = np.arange(100)
dropList = random.sample(range(0,99),(100 - ensemble_size))
trainList = np.array([0,1,2,3,4,5,6,8,9,10,11,12,13,14,16,17,18,19,20,21])#np.delete(allList, dropList)
print("Train list: ", trainList)

# The data is read by the readData.py script. The data is appended to the list so
# The final shape of l is Ensemble_Size x Num_Frames x num_atoms x 6
l = []
l, nd_all, nd_temp2 = readData(l, trainList, trunc_start, trunc_stop)

# Intercept the List and concatenate the first 5 and last 5 atom ends to each frame


new_l = []
for item in l:
    new_item = np.concatenate((item, item[:,-5:,:]),axis=1)
    new_item = np.concatenate((item[:,:5,:], new_item),axis=1)
    new_item = np.concatenate((new_item, new_item[:,-5:,:]),axis=1)
    new_item = np.concatenate((new_item[:,:5,:], new_item),axis=1)
    #print(new_item.shape)
    new_l.append(new_item)

maxSeqlength=5
nFrames = l[0].shape[0]
print("Trunc size: ", nFrames)
NumberOfAtoms = l[0].shape[1]
lead_time = 15
print("Number of trajectories in ensemble: ", len(trainList))

# Here we create an instance of the ToySequenceData class
training_data = ToySequenceData(l, maxSeqlength, lead_time)
training_data.shuffle(seed = 0)


###
# The Model
###

### Defining the model
### Neural Architecture
# input 
# -> LSTM 10
# -> LSTM 32
# -> Add Positional Encoding X MLP
# -> LSTM 10
# -> take the last output
# -> MLP
# -> sigmoid

#     -
# - <   > - 
#     -

## input size of a single sequence object
input_size = 6
## intermediate size of the LSTM
hidden_size1 = 32
LSTM_numlayers = 1
output_size = 6 ## output size of a single object


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


    
def nll_constrained_gaussian(mu, sigma, y):
    mux = mu[:, 0]
    muy = mu[:, 1]
    muz = mu[:, 2]

    stdx = sigma[:, 0]
    stdy = sigma[:, 1]
    stdz = sigma[:, 2]

    truex = y[:, 0]
    truey = y[:, 1]
    truez = y[:, 2]

    squarex = (mux - truex) ** 2
    msx = squarex / (stdx**2) + pt.log(stdx**2)

    squarey = (muy - truey) **2
    msy = squarey / (stdy**2) + pt.log(stdy**2)

    squarez = (muz - truez) **2
    msz = squarez / (stdz**2) + pt.log(stdz**2)
    
    minimize_this = pt.mean(msx) + pt.mean(msy) + pt.mean(msz)
    
    return minimize_this



####
# Training the Model
####



np.random.seed(123)
random.seed(123)
pt.manual_seed(123)

model = RNN(input_size, hidden_size1, LSTM_numlayers, output_size).to(device)

# Parameters
learning_rate = 0.0005
training_iters = 10000
batch_size = 5000
display_step = 20

## Display MSE Loss alongside the NLL Loss
mseloss = nn.MSELoss()
optimizer = pt.optim.Adam(params=model.parameters(), lr=learning_rate)


avg_loss = 0
model.double()
step = 1
while step < training_iters:
    batch_x, batch_y, seqlen, batch_id = training_data.next(batch_size)

    batch_x = batch_x.reshape(-1, maxSeqlength, input_size)
    torch_batch_x = pt.from_numpy(batch_x).double().to(device)

    batch_y = batch_y.reshape(batch_size, 3)
    torch_batch_y = pt.from_numpy(batch_y).double().to(device)

    mu, sigma = model(torch_batch_x)

    mse = mseloss(mu, torch_batch_y)
    loss = nll_constrained_gaussian(mu, sigma, torch_batch_y)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    avg_loss += loss.item()
    

    if (step+1) % display_step == 0:
        print ('step {}, Loss: {}, mseloss: {}' 
            .format(step, avg_loss / display_step, mse.item()))
        
        avg_loss = 0
    step += 1

###
# Saving the Model
###
PATH = "./pt_trained_models/v1.pt"
pt.save(model.state_dict(), PATH)

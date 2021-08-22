import sys
import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from os import listdir
from os.path import isfile, join
import numpy
from bisect import bisect
from random import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
from random import random
import collections
import time
import codecs
import json
from numpy import mean, sqrt, square, arange
from sys import argv
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


def Loop_round2_newseed(wround, lead_time):
    # Seeding event
    ground_truth = np.load("seed.npy")

    #start = wround * 10
    #stop = start + lead_time
    print("New bucket in wround ", wround)
    #print("Seeding from frame ", start, " to frame ", stop)

    Coordinates = ground_truth[:15]

    # This has to change later
#     nd_temp, nd_temp2 = readSeed_save(wround)
    return Coordinates
    
    
######################################################################

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
    
    
                                              
# Again, same as before

# Parameters
learning_rate = 0.01
training_iters = 2000
batch_size = 5000
display_step = 10

# number of units in RNN cell
n_hidden = 32

def dynamicRNN(x, seqlen, weights, biases):
    
    print(x)
    x = tf.unstack(x, maxSeqlength, 1)
    print(x)
    
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)

    print(outputs)
    
    outputs = tf.stack(outputs)
    print(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])
    print(outputs)

    batch_size = tf.shape(outputs)[0]
    index = tf.range(0, batch_size) * maxSeqlength + (seqlen - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
    print(outputs)

    return tf.add(tf.matmul(outputs, weights['out']),biases['out'])
    # outputs is hidden layer output. This is being multiplied by weights. Then biases is being added.


maxSeqlength = 5

# wround is the round number. Can go from 0 to whatever
# For this code, wround has to be 0
# Note that round is not the same as chunk/window

# wround = int(argv[1])
# lambd = float(argv[2])
wround = int(0)
# lambd = 0.5
# chunk = 4
# outName = "generated_chunks/test4l5.xyz"
train_model_folder = "trained_models/"

chunk = int(argv[1])
lambd = float(argv[2])
outName = float(argv[3])

print("Generation Lambda = ", lambd)

# Based on the round, we decide which chunk we are in.
# This is opposite of what getBucket was doing for us before
# print(wround)
# chunk = getChunk(wround)

trunc_start, trunc_stop = getBucket(chunk)

lead_time = 15

if wround == 0:
    nd_temp, nd_temp2 = readSeed_save(wround)
    print("Seed saved after normalization")
    Coordinates  = readSeed_round0()
elif wround == 1:
    Coordinates, nd_temp, nd_temp2 = readSeed_round1(wround, chunk)
else:
    chunk_2 = getChunk(wround-2)
    chunk_1 = getChunk(wround-1)
    if chunk == chunk_1:
        Coordinates, nd_temp, nd_temp2 = read_round2_noseed(wround, chunk_2, chunk_1)
    else:
        nd_temp, nd_temp2 = readSeed_save(wround)
        print("New seed saved after normalization, for bucket ", str(chunk))
        Coordinates, nd_temp, nd_temp2 = read_round2_newseed(wround, lead_time)

NumberOfAtoms = Coordinates.shape[1]
print("Coordinates shape: ", Coordinates.shape)
 
print("Round: ", wround)
print("Choosing bucket ", chunk)
print(len(Coordinates), len(Coordinates[0]))


################################################################################


## seed 0
print("wround: ", wround)
print("Seed 0")
nd_temp, nd_temp2 = readSeed_save(wround)
print("Seed saved after normalization")
Coordinates  = readSeed_round0()

training_data = ToySequenceData(Coordinates, maxSeqlength, lead_time)

print("len of training_data.data: ", len(training_data.data))
print("len of training_data.data[0]: ", len(training_data.data[0]))

all_x, all_y, all_y_previous, all_seqlen, batch_id = training_data.getAll()
np_all_y=np.array(all_y)

np_all_y_previous=np.array(all_y_previous)


# In each round, we predict 10 frames. This is related to maxSeqlength and lead_time.
# I have forgotten the exact relationship, but something simple.
# Maybe (lead_time - maxSeqlength) is the number of frames we can predict in each round
nframes_pred = 10

fName = "../11_forcedcd/00/backbone.npy"
rawCood_temp = np.load(fName)
predicted_frames_real_space = np.zeros((1,40,3)) ## just to allow concatenation
scaled_pred_frames = np.zeros((1,40,3))


model_path = train_model_folder + 'my-model-6D-' + str(chunk) + '-10000'


with tf.Session() as sess:
    saver = tf.train.import_meta_graph(model_path + ".meta")
#     saver.restore(sess,tf.train.latest_checkpoint('test_ch6/'))
    load_path = saver.restore(sess, model_path)


    NumberOfAtoms=sess.run('NumberOfAtoms:0')
    maxSeqlength=sess.run('maxSeqlength:0')
    n_hidden=sess.run('n_hidden:0')

    graph=tf.get_default_graph()

    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    seqlen = graph.get_tensor_by_name("seqlen:0")

#     pred = graph.get_tensor_by_name("pred:0")
    dist = graph.get_tensor_by_name("dist:0")
    
    if(chunk != 1):
        print("Seeding NEW CHUNK")
        nd_temp, nd_temp2 = readSeed_save(wround = chunk * 97)
        Coordinates = Loop_round2_newseed(wround, lead_time)
    
    wround = 0
    while wround < 97:
        
        print(wround)
        ### Coordinates is in scaled space remember
        training_data = ToySequenceData(Coordinates, maxSeqlength, lead_time)

        all_x, all_y, all_y_previous, all_seqlen, batch_id = training_data.getAll()
        distribution = sess.run(dist, feed_dict={x: all_x, y: all_y, seqlen: all_seqlen})

        pred_x_all = tf.convert_to_tensor(distribution[:,0] + (np.random.normal(0, 1, len(all_x)) * np.exp(distribution[:,3]) * lambd))
        pred_y_all=tf.convert_to_tensor(distribution[:,1] + (np.random.normal(0, 1, len(all_x)) * np.exp(distribution[:,4]) * lambd))
        pred_z_all=tf.convert_to_tensor(distribution[:,2] + (np.random.normal(0, 1, len(all_x)) * np.exp(distribution[:,5]) * lambd))
        my_predictions_raw=tf.transpose(tf.concat([[pred_x_all,pred_y_all,pred_z_all]],0),name='my_predictions_raw')

        # mpr is the prediction in the scaled space.
        # we need to rescale it
        mpr = my_predictions_raw.eval()
        temp = np.reshape(mpr, (nframes_pred, NumberOfAtoms,3))

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
        
#         print(Coordinates[0][0])
    
    

    # my_predictions is the prediction in real dimensions
#     my_predictions = np.reshape(new.inversed_dat, (nframes_pred * NumberOfAtoms,3))

#     with open("Generation_round" + str(wround) + "_" + str(lead_time) + "_" + str(n_hidden) + "_" + str(maxSeqlength) + "_" + str(chunk) + ".txt", 'w') as outputfile2:
#         with open("Generation_round" + str(wround) + "_unscaled.txt", 'w') as outputfile:
#             outputfile2.write("Lead time: " + str(lead_time) + "\n")    
#             outputfile2.write("Hidden units: " + str(n_hidden) + "\n")  
#             outputfile2.write("History: " + str(maxSeqlength) + "\n\n")

#             #my_predictions_raw = sess.run(pred, feed_dict={x: all_x, seqlen: all_seqlen})
#             distribution = sess.run(dist, feed_dict={x: all_x, y: all_y, seqlen: all_seqlen})



#             # mpr is the prediction in the scaled space.
#             # we need to rescale it
#             mpr = my_predictions_raw.eval()
#             temp = np.reshape(mpr, (nframes_pred, NumberOfAtoms,3))

#             # This is where we invoke the inversedData class
#             new = inversedData(nd_temp)
#             new.inverseIt(temp)
#             h = new.inversed_dat
            
#             # my_predictions is the prediction in real dimensions
#             my_predictions = np.reshape(new.inversed_dat, (nframes_pred * NumberOfAtoms,3))
#             # Now just write them out
#             for i in range(len(all_x)):
#                 outputfile2.write('frame ' + str(int(i/NumberOfAtoms) + maxSeqlength + lead_time) + ', atom ' + str(int(i%NumberOfAtoms)) + ', '  + str(my_predictions[i][0]) + ',' + str(my_predictions[i][1]) + ',' + str(my_predictions[i][2]) + '\n')

#                 outputfile.write('frame ' + str(int(i/NumberOfAtoms) + maxSeqlength + lead_time) + ', atom ' + str(int(i%NumberOfAtoms)) + ', '  + str(mpr[i][0]) + ',' + str(mpr[i][1]) + ',' + str(mpr[i][2]) + '\n')


###### WRITE OUT THE PREDICTIONS

all_predicted_frames = predicted_frames_real_space[1:]

my_predictions = all_predicted_frames

frame_num = my_predictions.shape[0]

nAtoms = "40"
with open(outName, "w") as outputfile:
    for frame_idx in range(frame_num):
        
        frame = my_predictions[frame_idx]
        outputfile.write(str(nAtoms) + "\n")
        outputfile.write(" generated by Chitrak\n")

        atomType = "CA"
        for i in range(40):
            line = str(frame[i][0]) + " " + str(frame[i][1]) + " " + str(frame[i][2]) + " "
            line += "\n"
            outputfile.write("  " + atomType + "\t" + line)

print("=> Finished Generation <=")

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

sys.argv = ['',0]

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"
    

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

wround = int(argv[1])
lambd = float(argv[2])
# bucket_frame_size = int(argv[3])
# seed_noseed = int(argv[4])
bucket_frame_size = int(0)
seed_noseed = int(0) ## seed=0, noseed=1

print("Generation Lambda = ", lambd)
lambd = 0.025
print("Generation Lambda = ", lambd)

# Based on the round, we decide which chunk we are in.
# This is opposite of what getBucket was doing for us before
print(wround)
chunk = getChunk(wround)

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
        
        if seed_noseed == 1: ## no seed
            Coordinates, nd_temp, nd_temp2 = read_round2_noseed(wround, chunk_2, chunk_1)
        else: ## seeding
            nd_temp, nd_temp2 = readSeed_save(wround)
            print("New seed saved after normalization, for bucket ", str(chunk))
            Coordinates, nd_temp, nd_temp2 = read_round2_newseed(wround, lead_time)

NumberOfAtoms = Coordinates.shape[1]
print("Coordinates shape: ", Coordinates.shape)
 
print("Round: ", wround)
print("Choosing bucket ", chunk)
print(len(Coordinates), len(Coordinates[0]))



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
#n_hidden_temp = 32

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./my-model-6D-' + str(chunk) + '-10000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))

    NumberOfAtoms=sess.run('NumberOfAtoms:0')
    maxSeqlength=sess.run('maxSeqlength:0')
    n_hidden=sess.run('n_hidden:0')

    graph=tf.get_default_graph()

    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    seqlen = graph.get_tensor_by_name("seqlen:0")

    #pred = graph.get_tensor_by_name("pred:0")
    dist = graph.get_tensor_by_name("dist:0")
    
    with open("Generation_round" + str(wround) + "_" + str(lead_time) + "_" + str(n_hidden) + "_" + str(maxSeqlength) + "_" + str(chunk) + ".txt", 'w') as outputfile2:
        with open("Generation_round" + str(wround) + "_unscaled.txt", 'w') as outputfile:
            outputfile2.write("Lead time: " + str(lead_time) + "\n")    
            outputfile2.write("Hidden units: " + str(n_hidden) + "\n")  
            outputfile2.write("History: " + str(maxSeqlength) + "\n\n")

            #my_predictions_raw = sess.run(pred, feed_dict={x: all_x, seqlen: all_seqlen})
            distribution = sess.run(dist, feed_dict={x: all_x, y: all_y, seqlen: all_seqlen})

            pred_x_all = tf.convert_to_tensor(distribution[:,0] + (np.random.normal(0, 1, len(all_x)) * distribution[:,3] * lambd))
            pred_y_all=tf.convert_to_tensor(distribution[:,1] + (np.random.normal(0, 1, len(all_x)) * distribution[:,4] * lambd))
            pred_z_all=tf.convert_to_tensor(distribution[:,2] + (np.random.normal(0, 1, len(all_x)) * distribution[:,5] * lambd))
            my_predictions_raw=tf.transpose(tf.concat([[pred_x_all,pred_y_all,pred_z_all]],0),name='my_predictions_raw')

            # mpr is the prediction in the scaled space.
            # we need to rescale it
            mpr = my_predictions_raw.eval()
            temp = np.reshape(mpr, (nframes_pred, NumberOfAtoms,3))

            # This is where we invoke the inversedData class
            new = inversedData(nd_temp)
            new.inverseIt(temp)
            h = new.inversed_dat
            
            # my_predictions is the prediction in real dimensions
            my_predictions = np.reshape(new.inversed_dat, (nframes_pred * NumberOfAtoms,3))
            # Now just write them out
            for i in range(len(all_x)):
                outputfile2.write('frame ' + str(int(i/NumberOfAtoms) + maxSeqlength + lead_time) + ', atom ' + str(int(i%NumberOfAtoms)) + ', '  + str(my_predictions[i][0]) + ',' + str(my_predictions[i][1]) + ',' + str(my_predictions[i][2]) + '\n')

                outputfile.write('frame ' + str(int(i/NumberOfAtoms) + maxSeqlength + lead_time) + ', atom ' + str(int(i%NumberOfAtoms)) + ', '  + str(mpr[i][0]) + ',' + str(mpr[i][1]) + ',' + str(mpr[i][2]) + '\n')
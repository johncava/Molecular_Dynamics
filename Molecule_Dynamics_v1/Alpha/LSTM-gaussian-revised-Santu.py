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
import collections
import time
import codecs
import json
from numpy import mean, sqrt, square, arange
from readData import *
from getBucket import getBucket
from sys import argv
 
start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"
    

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
                                              
        return batch_data, batch_labels, batch_seqlen,self.batch_id


    def getAll(self):                                                                                        
        return self.data, self.labels, self.y_previous,self.seqlen,self.batch_id
    
                                               
# These are some of our hyperparameters.
# For now, we won't change these.

# Parameters
learning_rate = 0.0005
training_iters = 10000
batch_size = 5000
display_step = 20

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

# "chunk" is basically the window (splitting up the trajectory into
# multiple parts). Note that the terms "chunk" and "bucket" are used
# interchangeably here.
chunk = float(argv[1])

# getBucket decides which part of the trajectory is used for learning.
# For chunk 1, it is frames 0 to 1000. For chunk 2, it is 980 to 2000, etc.
# The extra 20 frames come in because maxSeqlength = 5 and lead_time = 15.
trunc_start, trunc_stop = getBucket(chunk)
print("Using frames: ", trunc_start, " to: ", trunc_stop)

# This is ensemble size. This is one of the parameters that we need to change
ensemble_size = 10

# The next few lines create a random list of numbers between 0 and 99.
# Size of the list is ensemble_size. This list determines which trajectories
# get chosen for training.

# When you do the training multiple times, each time a separate set of
# trajectories get selected.

allList = np.arange(100)
dropList = random.sample(range(0,99),(100 - ensemble_size))
trainList = np.delete(allList, dropList)
print("Train list: ", trainList)


# The data is read by the readData.py script. Will explain it separately.
# For now, the "Coordinates 
l = []
l, nd_all, nd_temp2 = readData(l, trainList, trunc_start, trunc_stop)

maxSeqlength=5
nFrames = l[0].shape[0]
print("Trunc size: ", nFrames)
NumberOfAtoms = l[0].shape[1]
lead_time = 15
print("Number of trajectories in ensemble: ", len(trainList))


# Here we create an instance of the ToySequenceData class
training_data = ToySequenceData(l, maxSeqlength, lead_time)

print("Length of training data: ", len(training_data.data))
print("Length of first element of training_data: ", len(training_data.data[0]))

# Our input is 6D, but output is 3D. We are only interested in
# predicting the (X,Y,Z) values. Rest we can calculate.

# tf Graph input
x = tf.placeholder("float", [None, maxSeqlength, 6],name='x')
y = tf.placeholder("float", [None, 3],name='y')
seqlen = tf.placeholder(tf.int32, [None],name='seqlen')
tf.Variable(NumberOfAtoms,name='NumberOfAtoms')
tf.Variable(maxSeqlength,name='maxSeqlength')
tf.Variable(n_hidden,name='n_hidden')

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, 6]),name='weights')
}
biases = {
    'out': tf.Variable(tf.random_normal([6]),name='biases')
}


print("before pred")
distribution = tf.identity(dynamicRNN(x,seqlen,weights, biases),name="dist")

# This is where noise gets added
pred_x=distribution[:,0] + (np.random.normal(0, 1, batch_size) * distribution[:,1])
pred_y=distribution[:,2] + (np.random.normal(0, 1, batch_size) * distribution[:,3])
pred_z=distribution[:,4] + (np.random.normal(0, 1, batch_size) * distribution[:,5])

pred=tf.transpose(tf.concat([[pred_x],[pred_y],[pred_z]],0),name='pred')
print(pred)
print("after pred")

# This is where noise gets added
all_x, all_y, all_y_previous, all_seqlen, batch_id = training_data.getAll()
pred_x_all=distribution[:,0] + (np.random.normal(0, 1, len(all_x)) * distribution[:,1])
pred_y_all=distribution[:,2] + (np.random.normal(0, 1, len(all_x)) * distribution[:,3])
pred_z_all=distribution[:,4] + (np.random.normal(0, 1, len(all_x)) * distribution[:,5])

pred_all=tf.transpose(tf.concat([[pred_x_all],[pred_y_all],[pred_z_all]],0),name='pred_all')

# Loss and optimizer
Errors = tf.losses.mean_squared_error(predictions=pred, labels=y)
cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
accuracy = tf.reduce_mean(tf.cast(cost, tf.float32),name='accuracy')

# Initializing the variables
init = tf.global_variables_initializer()

np_all_x = np.array(all_x)
np_all_y = np.array(all_y)
np_all_y_previous = np.array(all_y_previous)
np_all_seqlen = np.array(all_seqlen)

print("all_x shape: ", np_all_x.shape)
print("all_y shape: ", np_all_y.shape)
print("all_y_previous shape: ", np_all_y_previous.shape)
print("all_seqlen shape: ", np_all_seqlen.shape)

# Now training begins
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    step = 1
    # Keep training until reach max iterations
    while step < training_iters:
        batch_x, batch_y, batch_seqlen,batch_id = training_data.next(batch_size)
        
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       seqlen: batch_seqlen})
        if step % display_step == 0:
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,
                                             seqlen: batch_seqlen})
            print("Iter " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss))
        

        if step % 1000 == 0:
            with open("Predictions.txt", 'w') as outputfile2:
                with open("output.txt", 'w') as outputfile:
                    my_predictions=sess.run(pred_all, feed_dict={x: all_x, y: all_y, seqlen: all_seqlen})

                    # This step writes some outputs, but honestly, we don't need these.
                    # Notice that these outputs are in the scaled space (0,1) and (-1,1).
                    # They need to be converted to real scale. I didn't do it here because
                    # we don't really care for this output.

                    # However, if the standard deviation needs to be written out, I am guessing
                    # this will be the step for it. If you can find a way of writing it, I know how
                    # to scale it back to the real space (will share eventually)

                    for i in range(len(all_x)):
                        outputfile2.write('frame ' + str(int(i/NumberOfAtoms)+maxSeqlength) + ', atom '+ str(int(i%NumberOfAtoms))+', ' +str(my_predictions[i][0])+ ','+ str(my_predictions[i][1])+','+ str(my_predictions[i][2])+'\n')

                    loss = sqrt(mean(square(my_predictions - (np_all_y))))

                    Naive_loss=sqrt(mean(square(np_all_y_previous - (np_all_y))))


                    print("Overall Status in Step: " + str(step) + ",Naive Loss= " + \
                      "{:.6f}".format(Naive_loss)+", Loss= " + \
                      "{:.6f}".format(loss)+'\n\n')
                             

        step += 1
        
    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))

    saver.save(sess, './my-model-6D-' + str(int(chunk)), global_step=10000)






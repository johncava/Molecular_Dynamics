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
    
           
##############

### Preparing the model, setting up the model 
# tf.reset_default_graph()

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
# chunk = float(4)
train_model_folder = "trained_models"
# This is ensemble size. This is one of the parameters that we need to change
ensemble_size = float(argv[2])
# ensemble_size = 10


# getBucket decides which part of the trajectory is used for learning.
# For chunk 1, it is frames 0 to 1000. For chunk 2, it is 980 to 2000, etc.
# The extra 20 frames come in because maxSeqlength = 5 and lead_time = 15.
trunc_start, trunc_stop = getBucket(chunk)
print("Using frames: ", trunc_start, " to: ", trunc_stop)


# The next few lines create a random list of numbers between 0 and 99.
# Size of the list is ensemble_size. This list determines which trajectories
# get chosen for training.

# When you do the training multiple times, each time a separate set of
# trajectories get selected.

allList = np.arange(100)
dropList = random.sample(range(0,99),(100 - ensemble_size))
trainList = np.array([20,21,36,42,56,59,60,69,76,99])#p.delete(allList, dropList)
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

############################ Loss Function

## negative log likelihood to learn the individual molecular trajectories
## the constraint that it is a ball, only learning the cov(x,x), cov(y,y) cov(z,z)
def nll_constrained_multivariate(dist, y):
    
    ## Exctracting the predicted mean and the predicted std in XYZ
    mu_x_preclip = dist[:,0] # shape (batchsize,)
    mu_y_preclip = dist[:,1] # shape (batchsize,)
    mu_z_preclip = dist[:,2] # shape (batchsize,)
    sigma_x_preclip = dist[:,3] # shape (batchsize,)
    sigma_y_preclip = dist[:,4] # shape (batchsize,)
    sigma_z_preclip = dist[:,5] # shape (batchsize,)
    
    true_x = y[:, 0]
    true_y = y[:, 1]
    true_z = y[:, 2]
    
    ## making sure the mean values are constrained between 0 and 1
    mu_x = tf.clip_by_value(t= mu_x_preclip, clip_value_min=tf.constant(1E-4), clip_value_max=tf.constant(1E+0)) # shape (batchsize,)
    mu_y = tf.clip_by_value(t= mu_y_preclip, clip_value_min=tf.constant(1E-4), clip_value_max=tf.constant(1E+0)) # shape (batchsize,)
    mu_z = tf.clip_by_value(t= mu_z_preclip, clip_value_min=tf.constant(1E-4), clip_value_max=tf.constant(1E+0)) # shape (batchsize,)

    ## making sure the sigma values are positive
    sigma_x = tf.clip_by_value(t=tf.exp(sigma_x_preclip), clip_value_min=tf.constant(1E-4), clip_value_max=tf.constant(1E+100)) # shape (batchsize,)
    sigma_y = tf.clip_by_value(t=tf.exp(sigma_y_preclip), clip_value_min=tf.constant(1E-4), clip_value_max=tf.constant(1E+100)) # shape (batchsize,)
    sigma_z = tf.clip_by_value(t=tf.exp(sigma_z_preclip), clip_value_min=tf.constant(1E-4), clip_value_max=tf.constant(1E+100)) # shape (batchsize,)

    
    
    ## element wise square
    square_x = tf.square(mu_x - true_x)
    sigma_x_squared = tf.square(sigma_x)
    msx = tf.add(tf.divide(square_x,sigma_x_squared), tf.log(sigma_x_squared))
    
    ## element wise square
    square_y = tf.square(mu_y - true_y)
    sigma_y_squared = tf.square(sigma_y)
    msy = tf.add(tf.divide(square_y,sigma_y_squared), tf.log(sigma_y_squared))
    
    ## element wise square
    square_z = tf.square(mu_z - true_z)
    sigma_z_squared = tf.square(sigma_z)
    msz = tf.add(tf.divide(square_z,sigma_z_squared), tf.log(sigma_z_squared))

    minimize_this = tf.reduce_mean(msx) + tf.reduce_mean(msy) + tf.reduce_mean(msz)
    return(minimize_this)

############################################################################################################

# This is where noise gets added
# pred_x=distribution[:,0] + (np.random.normal(0, 1, batch_size) * distribution[:,1])
# pred_y=distribution[:,2] + (np.random.normal(0, 1, batch_size) * distribution[:,3])
# pred_z=distribution[:,4] + (np.random.normal(0, 1, batch_size) * distribution[:,5])

# pred=tf.transpose(tf.concat([[pred_x],[pred_y],[pred_z]],0),name='pred')
# print(pred)
# print("after pred")

# This is where noise gets added
# all_x, all_y, all_y_previous, all_seqlen, batch_id = training_data.getAll()
# pred_x_all=distribution[:,0] + (np.random.normal(0, 1, len(all_x)) * distribution[:,1])
# pred_y_all=distribution[:,2] + (np.random.normal(0, 1, len(all_x)) * distribution[:,3])
# pred_z_all=distribution[:,4] + (np.random.normal(0, 1, len(all_x)) * distribution[:,5])

# pred_all=tf.transpose(tf.concat([[pred_x_all],[pred_y_all],[pred_z_all]],0),name='pred_all')

# Loss and optimizer
# Errors = tf.losses.mean_squared_error(predictions=pred, labels=y)
# cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions=pred, labels=y))
cost = nll_constrained_multivariate(distribution, y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
accuracy = tf.reduce_mean(tf.cast(cost, tf.float32),name='accuracy')


################################################################################################################################################


##just to have here in case I want to use
def MSE_error_function(pred_means, y):
    return tf.reduce_mean(tf.losses.mean_squared_error(predictions=pred_means, labels=y))

## NOT training using MSE, but having it here to view during training
MSE_loss_op = MSE_error_function(distribution[:,:3], y)

# Initializing the variables
init = tf.global_variables_initializer()

# np_all_x = np.array(all_x)
# np_all_y = np.array(all_y)
# np_all_y_previous = np.array(all_y_previous)
# np_all_seqlen = np.array(all_seqlen)

# print("all_x shape: ", np_all_x.shape)
# print("all_y shape: ", np_all_y.shape)
# print("all_y_previous shape: ", np_all_y_previous.shape)
# print("all_seqlen shape: ", np_all_seqlen.shape)

### Added this outer for-loop to help with SGD convergence
max_tries = 100
num_tries = 1
reboot_check_step = 100


while(num_tries < max_tries):

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
                MSEloss = sess.run(MSE_loss_op, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
                NLLloss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
                print("Iter " + str(step) + ", Minibatch Loss= " + \
                          "{:.6f}".format(NLLloss) + " || " + "{:.6f}".format(MSEloss))

            if(step == reboot_check_step and MSEloss > 0.5):
                print("=> Rebooting SGD...")
                num_tries += 1
                break

            step += 1

            if(step == training_iters):
                print("Optimization Finished!")
                print("Elapsed time: ", elapsed(time.time() - start_time))

                saver.save(sess, train_model_folder + '/my-model-6D-' + str(int(chunk)), global_step=10000)

                num_tries += max_tries
                break

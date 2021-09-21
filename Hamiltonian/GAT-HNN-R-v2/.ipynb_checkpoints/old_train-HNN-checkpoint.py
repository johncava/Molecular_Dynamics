# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski
# Modified by Nick

import torch, argparse
import numpy as np
import pandas as pd
import time


import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(""))
THIS_DIR = os.path.join(THIS_DIR, "Mol-HNN-cuda-v3")
# PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(PARENT_DIR)

from cuda_nn_models import *
from cuda_hnn import HNN
from cuda_utils import L2_loss, to_pickle, from_pickle

from get_data import get_dataset, elapsed

import glob


####### DATA PREPARATION STEP. NOT NECESSARY AFTER THE FIRST RUN

##
# Loading in the whitened Dataset
##
num_trajectories = 200
num_atoms = 40
batch_size = 100
epochs = 3
save_every = 500 ## how often the model is saved


start_time = time.time()
# print("Elapsed time: ", elapsed(time.time() - start_time))



PATH = "models/uw-sept18-MOLHNNv3.pt"
raw_data = './../data/SMD_data/*.npy'
saved_x_dataset = "../data/uw_x_dataset.npy"
saved_dx_dataset = "../data/uw_dx_dataset.npy"
log_csv_name = "log.csv"


### Determining current 
if not os.path.isfile(log_csv_name):
    print("The log file does not exist, creating log file...")
    log_df = pd.DataFrame(columns = ['time', 'epoch', 'loss'])
    current_epoch = 0
else:
    print("Found the log file! Reading it...")
    log_df = pd.read_csv(log_csv_name)
    current_epoch = log_df.iloc[-1]['epoch']
    print("Current Epoch is: ", current_epoch)

    
def add_log_csv(log_df, log_csv_name, start_time, epoch, loss):
    time_elapsed = elapsed(time.time() - start_time)
    next_step = pd.DataFrame([{'time': time_elapsed, 'epoch': epoch, 'loss': loss}])
    log_df = log_df.append(next_step)
    log_df.to_csv(log_csv_name)
    return log_df
    

######## LOADING THE DATA
if os.path.isfile(saved_x_dataset) and os.path.isfile(saved_dx_dataset):
    print("Found data files => loaded in the old data at:", saved_x_dataset)
    x_dataset = np.load(saved_x_dataset)
    dx_dataset = np.load(saved_dx_dataset)
    
else:
    print("Can't find data files => Creating a new set")
    x_dataset, dx_dataset = get_dataset(raw_data, saved_x_dataset, saved_dx_dataset, num_trajectories, num_atoms)

x = torch.tensor(x_dataset, requires_grad=True, dtype=torch.float32).cuda()
dxdt = torch.Tensor(dx_dataset).cuda()

epoch_steps = int(x_dataset.shape[0] / batch_size)
# total_steps = int(epoch_steps * epochs)


#### LOADING THE PARAMETERS

# arrange data
def get_args():
    return {'input_dim': 240, # 40 atoms, each with q_x, q_y, p_z, p_y
         'hidden_dim': 200,
         'learn_rate': 1e-3,
         'input_noise': 0.1, ## NO INPUT NOISE YET
         'batch_size': 100,
         'nonlinearity': 'leaky',
#          'total_steps': total_steps, ## 3 epochs effectively i guess
         'field_type': 'solenoidal', ## change this? solenoidal
         'print_every': 200,
         'verbose': True,
         'name': '2body',
         'baseline' : False,
         'seed': 0,
         'save_dir': '{}'.format(THIS_DIR),
         'fig_dir': './figures'}


class ObjectView(object):
    def __init__(self, d): self.__dict__ = d

args = ObjectView(get_args())



#### THIS IS WHERE YOU INTERCEPT IF YOU WANT TO USE A DIFFERENT MODEL
output_dim = 2
# nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
num_particles = 40
channel_size = 3
hidden_size = 32
output_size = 2
nn_model = GATModel(channel_size, hidden_size, output_size).cuda()

model = HNN(args.input_dim, differentiable_model=nn_model,
        field_type=args.field_type, baseline=args.baseline)
optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=0)


## Check if the model exists at this stage right here:
if os.path.isfile(PATH):
    model.load_state_dict(torch.load(PATH))


### Training the data

# vanilla train loop
stats = {'train_loss': [], 'test_loss': []}

for epoch in range(epochs):
    for step in range(epoch_steps):

        # train step
        ixs = torch.randperm(x.shape[0])[:args.batch_size]
        dxdt_hat = model.time_derivative(x[ixs])
        dxdt_hat += args.input_noise * torch.randn(*x[ixs].shape).cuda() # add noise, maybe

        loss = L2_loss(dxdt[ixs], dxdt_hat)
        loss.backward()
    #     grad = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
        optim.step() ; optim.zero_grad()

        # run test data
    #     test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
    #     test_dxdt_hat = model.time_derivative(test_x[test_ixs])
    #     test_dxdt_hat += args.input_noise * torch.randn(*test_x[test_ixs].shape) # add noise, maybe
    #     test_loss = L2_loss(test_dxdt[test_ixs], test_dxdt_hat)


        # logging
        stats['train_loss'].append(loss.item())
    #     stats['test_loss'].append(test_loss.item())
        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}"
              .format(step, loss.item()))

        if step % save_every == 0:
            torch.save(model.state_dict(), PATH) ### SAVING THE MODEL EVERY FEW STEPS
            write_epoch = current_epoch + epoch ## this gets updated everytime the file terminates
            add_log_csv(log_df, log_csv_name, start_time, write_epoch, loss.item())

print("=> Finished Training <=")


## Saving model
torch.save(model.state_dict(), PATH) ### SAVING THE MODEL EVERY FEW STEPS
write_epoch = current_epoch + epoch ## this gets updated everytime the file terminates
add_log_csv(log_df, log_csv_name, start_time, write_epoch, loss.item())





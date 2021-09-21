#####
## This version has two main edits
## 1) downsampled to much fewer frames
## 2) recurrent training 

import torch, argparse
import numpy as np
import pandas as pd
import time


import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(""))
THIS_DIR = os.path.join(THIS_DIR, "GAT-HNN-R-v2")
# PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(PARENT_DIR)

from cuda_nn_models import *
from cuda_hnn import HNN
from cuda_utils import L2_loss, to_pickle, from_pickle
from get_data import elapsed
from get_data import get_dataset ## we dont need this now
import glob


##
# Loading in the whitened Dataset
##
num_atoms = 40
batch_size = 100
epochs = 3
save_every = 500 ## how often the model is saved


start_time = time.time()
# print("Elapsed time: ", elapsed(time.time() - start_time))


PATH = "models/GATHNNRv1.pt"
raw_data = './../data/SMD_data/*.npy'
saved_x_dataset = "../data/ds_250_uw_x_dataset.npy"
saved_dx_dataset = "../data/ds_250_uw_dx_dataset.npy"
downsample_num = 80
seq_len = 10
log_csv_name = "log.csv"

data_whitened = False


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
    log_df.to_csv(log_csv_name, index=False)
    return log_df



######## LOADING THE DATA
if os.path.isfile(saved_x_dataset) and os.path.isfile(saved_dx_dataset):
    print("Found data files => loaded in the old data at:", saved_x_dataset)
    x_dataset = np.load(saved_x_dataset)
    dx_dataset = np.load(saved_dx_dataset)
    
else:
    print("Can't find data files => Creating a new set")
    x_dataset, dx_dataset = get_dataset(raw_data, saved_x_dataset, saved_dx_dataset, num_atoms, downsample_num, data_whitened)
    


    

## Now we have to process these into sequences of 10 or whatever you want
num_traj = x_dataset.shape[0]
num_points = x_dataset.shape[1]

# dataset = []
x_dat = []
dx_seq_dat = []
for traj in range(num_traj):
    x = x_dataset[traj]
    dx = dx_dataset[traj]
    
    for i in range(num_points - seq_len):
        x_dat.append(x[i])
        dx_seq_dat.append(dx[i:i+seq_len])
#         position = torch.tensor(x[i], requires_grad=True, dtype=torch.float32).cuda()
#         momenta_sequence = torch.Tensor(dx[i:i+seq_len]).cuda()
#         dataset.append((position,momenta_sequence ))

x_dat = np.array(x_dat)
dx_seq_dat = np.array(dx_seq_dat)

x_dat = torch.tensor(x_dat, requires_grad=True, dtype=torch.float32).cuda()
dx_seq_dat = torch.Tensor(dx_seq_dat).cuda()

print("current x_dataset size: ", x_dat.shape)
print("current dx_dataset size: ", dx_seq_dat.shape)


##### Setting up the model

## Now creating the model
#### LOADING THE PARAMETERS

# arrange data
def get_args():
    return {'input_dim': 240, # 40 atoms, each with q_x, q_y, p_z, p_y
         'hidden_dim': 200,
         'learn_rate': 1e-3,
         'input_noise': 0., ## NO INPUT NOISE YET
         'batch_size': 100,
         'nonlinearity': 'softplus',
#          'total_steps': total_steps, ## 3 epochs effectively i guess
         'field_type': 'helmholtz', ## change this? solenoidal
         'print_every': 5,
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
hidden_size = 64
output_size = 2
nn_model = GATModel(channel_size, hidden_size, output_size).cuda()

model = HNN(args.input_dim, differentiable_model=nn_model,
        field_type=args.field_type, baseline=args.baseline)
optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=0)


## Check if the model exists at this stage right here:
if os.path.isfile(PATH):
    model.load_state_dict(torch.load(PATH))


##### TRAINING
# vanilla train loop
stats = {'train_loss': [], 'test_loss': []}

batch_size = 10

for epoch in range(epochs):
    for step in range(int(x_dat.shape[0] / batch_size)):
        
        
        ixs = torch.randperm(x.shape[0])[:batch_size]
        a = x_dat[ixs]
        dxdt = dx_seq_dat[ixs]
        traj_output = []
        for _ in range(seq_len):
            dxdt_hat = model.time_derivative(a)
            a = a + dxdt_hat
            traj_output.append(dxdt_hat)

        dxdt_hat = torch.stack(traj_output)
        dxdt_hat = torch.transpose(dxdt_hat, 0, 1)

        loss = L2_loss(dxdt, dxdt_hat)
        loss.backward()
        optim.step() ; optim.zero_grad()
        
#         print(step, " || ", loss.item())

        # logging
        stats['train_loss'].append(loss.item())
        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}"
              .format(step, loss.item()))
            
            f = open("trainlog.txt", "a")
            f.write("step {}, train_loss {:.4e}\n"
              .format(step, loss.item()))
            f.close()

        if step % save_every == 0 and step != 0:
            torch.save(model.state_dict(), PATH) ### SAVING THE MODEL EVERY FEW STEPS
            write_epoch = current_epoch + epoch ## this gets updated everytime the file terminates
            add_log_csv(log_df, log_csv_name, start_time, write_epoch, loss.item())

print("=> Finished Training <=")


## Saving model
torch.save(model.state_dict(), PATH) ### SAVING THE MODEL EVERY FEW STEPS
write_epoch = current_epoch + epoch ## this gets updated everytime the file terminates
add_log_csv(log_df, log_csv_name, start_time, write_epoch, loss.item())

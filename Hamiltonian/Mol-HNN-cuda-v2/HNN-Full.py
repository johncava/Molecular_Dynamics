# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski
# Modified by Nick

import torch, argparse
import numpy as np

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(""))
THIS_DIR = os.path.join(THIS_DIR, "Mol-HNN-cuda-v2")
# PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(PARENT_DIR)

from cuda_nn_models import MLP
from cuda_hnn import HNN
from cuda_utils import L2_loss, to_pickle, from_pickle


####### DATA PREPARATION STEP. NOT NECESSARY AFTER THE FIRST RUN

##
# Loading in the whitened Dataset
##
num_trajectories = 10
num_atoms = 40

import glob


files = glob.glob('./../../All_ML_Training_Data/210905_SMD_decaalanine/SMD/output/processed_orient/*.npy')
dataset = []
for num, file_ in enumerate(files):
    X_positions = np.load(file_)
    X = X_positions
    dataset.append(X)
    if len(dataset) == num_trajectories:
        break

dataset = np.array(dataset)
dataset = dataset.reshape(num_trajectories, -1, num_atoms*3)
print(dataset.shape)

### Preprocessing to create the training step

x_dataset = [] ## vector of position and momentum
dx_dataset = [] ## vector of change in position and momentum
## Getting the velocity of each trajectory
for traj_idx in range(num_trajectories):
    traj = dataset[traj_idx]
#     traj = dataset[0]
    num_datapoints = traj.shape[0] - 1

    momenta = []
    for i in range(traj.shape[0] - 1):
        dx = traj[i+1] - traj[i]
        momenta.append(dx)

    momenta = np.array(momenta) ## convert to an np array

    ## stack the coordinates so that we have all the positions and all the momentums
    coords = np.stack((traj[:-1], momenta), 1) ## shape: (19999, 2, 120)
    coords = coords.reshape(num_datapoints, 2 * 120) ## shape: (19999, 240)
    
    ## Now we need to calculate the change in coords (positions and momentums)
    delta_coords = []
    for frame in range(coords.shape[0] - 1):
        dx = coords[frame+1] - coords[frame]
        delta_coords.append(dx)
    delta_coords = np.array(delta_coords)
    
    ## appending to the dataset
    x_dataset.append(coords[:-1])
    dx_dataset.append(delta_coords)


x_dataset = np.array(x_dataset) ## shape of (100, 19998, 240)
dx_dataset = np.array(dx_dataset) ## shape of (100, 19998, 240)

x_dataset = x_dataset.reshape(-1, 240) ## shape of (1999800, 240)
dx_dataset = dx_dataset.reshape(-1, 240) ## shape of (1999800, 240)

'''
# x_dataset[0] + dx_dataset[0] == x_dataset[1] ## This is just a sanity check that we can predict the next step

## just for quick load don't need this

#np.save("whitened_x_dataset.npy", x_dataset)
#np.save("whitened_dx_dataset.npy", dx_dataset)


######## LOADING THE DATA

# x_dataset = np.load("whitened_x_dataset.npy")
# dx_dataset = np.load("whitened_dx_dataset.npy")

'''

x = torch.tensor(x_dataset, requires_grad=True, dtype=torch.float32).cuda()
dxdt = torch.Tensor(dx_dataset).cuda()

batch_size = 100
epochs = 3
total_steps = int(x_dataset.shape[0] / 100 * epochs)
total_steps



#### LOADING THE PARAMETERS

# arrange data
def get_args():
    return {'input_dim': 240, # 40 atoms, each with q_x, q_y, p_z, p_y
         'hidden_dim': 200,
         'learn_rate': 1e-3,
         'input_noise': 0.1, ## NO INPUT NOISE YET
         'batch_size': 100,
         'nonlinearity': 'leaky',
         'total_steps': total_steps, ## 3 epochs effectively i guess
         'field_type': 'helmholtz', ## change this? solenoidal
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
nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
model = HNN(args.input_dim, differentiable_model=nn_model,
        field_type=args.field_type, baseline=args.baseline)
optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=0)



### Training the data

# vanilla train loop
stats = {'train_loss': [], 'test_loss': []}
for step in range(args.total_steps+1):

    # train step
    ixs = torch.randperm(x.shape[0])[:args.batch_size]
    dxdt_hat = model.time_derivative(x[ixs])
    dxdt_hat += args.input_noise * torch.randn(*x[ixs].shape).cuda() # add noise, maybe
    
    loss = L2_loss(dxdt[ixs], dxdt_hat)
    loss.backward()
    grad = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
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
        print("step {}, train_loss {:.4e}, test_loss {:.4e}, grad norm {:.4e}, grad std {:.4e}"
          .format(step, loss.item(), 420, grad@grad, grad.std()))

print("=> Finished Training <=")



## Saving model
PATH = "white-MOLHNNv1.pt"
torch.save(model.state_dict(), PATH)




################
## Loading the Model
############

PATH = "white-MOLHNNv1.pt"
model.load_state_dict(torch.load(PATH))
model.eval()



### Autoregressive
initial_frame = x[0]
a = initial_frame
frames = []

for i in range(10000):
    dx_hat = model.time_derivative(a.reshape(-1, 240))
    a = a + dx_hat * 0.005
    a = a + torch.randn(a.shape) * 0.1 # add noise, maybe
    
    new_frame = a.detach().numpy()
    new_frame = new_frame[:, :120].reshape(40, 3) ## converts the frame of coordinates into XYZ
    frames.append(new_frame)

frames = np.array(frames)



# Save predictions into VMD format
predictions = frames
frame_num = predictions.shape[0]

nAtoms = "40"
outName = "HNNv2.xyz"
with open(outName, "w") as outputfile:
    for frame_idx in range(frame_num):
        
        frame = predictions[frame_idx]
        outputfile.write(str(nAtoms) + "\n")
        outputfile.write(" generated by JK\n")

        atomType = "CA"
        for i in range(40):
            line = str(frame[i][0]) + " " + str(frame[i][1]) + " " + str(frame[i][2]) + " "
            line += "\n"
            outputfile.write("  " + atomType + "\t" + line)

print("=> Finished Generation <=")
'''
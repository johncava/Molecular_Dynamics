# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch, argparse
import numpy as np

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(""))
THIS_DIR = os.path.join(THIS_DIR, "whitened-Mol-HNN")
# PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(PARENT_DIR)

from cuda_nn_models import MLP
from cuda_hnn import HNN
from cuda_utils import L2_loss, to_pickle, from_pickle
# from data import get_dataset, coords2state, get_orbit, random_config
# from data import potential_energy, kinetic_energy, total_energy


### Loading the Data
x_dataset = np.load("whitened_x_dataset.npy")
dx_dataset = np.load("whitened_dx_dataset.npy")
x = torch.tensor(x_dataset, requires_grad=True, dtype=torch.float32).cuda()
dxdt = torch.Tensor(dx_dataset).cuda()

batch_size = 100
epochs = 3
total_steps = int(x_dataset.shape[0] / 100 * epochs)



### Parameters
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



#### Model
output_dim = 2
nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
model = HNN(args.input_dim, differentiable_model=nn_model,
        field_type=args.field_type, baseline=args.baseline)
optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=0)



#### Training
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
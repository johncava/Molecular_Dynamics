import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

###
# Important Variables
###

number_of_particles = 40
input_size = 3
hidden_size = 128
history_size = 15
lead_time = 2
M = 5
num_layers = 1
batch_size = 128

##
# Read Dataset
##
import glob
import numpy as np

files = glob.glob('./../../All_ML_Training_Data/210905_SMD_decaalanine/SMD/output/processed_orient/*.npy')

dataset = []

end_to_end_distance = dict()
for i in range(100):
    end_to_end_distance[i] = []
    for j in range(int(40/2)):
        end_to_end_distance[i].append([])

for file_ in files:
    X_positions = np.load(file_)

    X = X_positions[:1000]

    X = X[::10]

    # Create Training dataset from this sequence
    #print(X.shape[0])-> 100
    for frame_num in range(X.shape[0]):
        dataset.append((frame_num, X[frame_num,:,:]))
        for j in range(int(40/2)):
            end_to_end_distance[frame_num][j].append(np.sqrt(np.power((X[frame_num,j,:] - X[frame_num,(40-1)-j,:]),2).sum()))

# Check the end to end distance per frame
for i in range(100):
    for j in range(int(40/2)):
        end_to_end_distance[i][j] = np.array(end_to_end_distance[i][j]).mean().tolist()
    #end_to_end_distance[i] = np.array(end_to_end_distance[i]).mean().tolist()


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.mlp1 = nn.Linear(32,50)
        self.mlp2 = nn.Linear(50,100)
        self.mlp3 = nn.Linear(100,120)

    def forward(self,batch_size, max_steps):
        t = random.choices(range(max_steps),k=batch_size)
        picked_t = t
        t = torch.tensor(t).view(batch_size,1)
        t = t/max_steps 
        t = t.float().cuda()
        z = torch.normal(0,1,size=(batch_size,31)).cuda()
        z = torch.cat((t,z),1)
        z = torch.sigmoid(self.mlp1(z))
        z = torch.sigmoid(self.mlp2(z))
        z = self.mlp3(z)
        return t, z, picked_t

    def generation_step(self, t, max_steps):
        t = torch.tensor(t).view(1,1)
        t = t/max_steps 
        t = t.float().cuda()
        z = torch.normal(0,1,size=(1,31)).cuda()
        z = torch.cat((t,z),1)
        z = torch.sigmoid(self.mlp1(z))
        z = torch.sigmoid(self.mlp2(z))
        z = self.mlp3(z)
        return z

# Case 1: Multiply Z with A
'''
mlp = torch.nn.Linear(20,40)
mlp2 = torch.nn.Conv1d(1,3,1)
z = torch.randn(batch_size,20)
z = mlp(z)
z = z.unsqueeze(-1).view(batch_size,1,40)
z = mlp2(z)
z = z.view(batch_size,40,3)
print(z.size())
x = []
for _ in range(number_of_particles):
    a = torch.normal(0,1,size=(batch_size,1,3))
    x.append(a)
x = torch.stack(x,1).squeeze(-2)
print(x.size())
new_x = z*x
print(new_x.size())
new_x = new_x.view(batch_size,-1)
print(new_x.size())
'''
# Case 2: Concatenate Z with A and MLP [Z,A]
mlp = torch.nn.Linear(20,40)
mlp2 = torch.nn.Conv1d(1,3,1)
mlp3 = torch.nn.Conv1d(6,3,1)
z = torch.randn(batch_size,20)
z = mlp(z)
z = z.unsqueeze(-1).view(batch_size,1,40)
z = mlp2(z)
z = z.view(batch_size,40,3)
print(z.size())
x = []
for _ in range(number_of_particles):
    a = torch.normal(0,1,size=(batch_size,1,3))
    x.append(a)
x = torch.stack(x,1).squeeze(-2)
print(x.size())
new_x = torch.cat([z,x],-1)
print(new_x.size())
new_x = mlp3(new_x.view(batch_size,6,40)).view(batch_size,40,3)
print(new_x.size())
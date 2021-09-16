import torch
from cuda_nn_models import *

x = torch.randn(100, 240).cuda() ## shape of (1999800, 240)
dx_dataset = torch.randn(100, 240) ## shape of (1999800, 240)

num_particles = 40
model = SchNet(num_particles).cuda()

output = model(x)
print(output.size())
import torch
from cuda_nn_models import *

x = torch.randn(100, 240).cuda() ## shape of (1999800, 240)
dx_dataset = torch.randn(100, 240) ## shape of (1999800, 240)

num_particles = 40
channel_size = 3
hidden_size = 32
output_size = 2
model = GATModel(channel_size, hidden_size, output_size).cuda()

output = model(x)
print(output.size())
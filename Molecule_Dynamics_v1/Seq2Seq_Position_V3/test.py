import torch

x = torch.randn((3,3))
y = x

n = x.size(0)
m = y.size(0)
d = x.size(1)

x = x.unsqueeze(1).expand(n, m, d)
y = y.unsqueeze(0).expand(n, m, d)

dist = torch.pow(x - y, 2).sum(2) 
print(dist)
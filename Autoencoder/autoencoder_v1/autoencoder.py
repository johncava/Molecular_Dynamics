import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 128
##
# Read Dataset
##
import glob
import numpy as np

files = glob.glob('./../../All_ML_Training_Data/210905_SMD_decaalanine/SMD/output/processed_orient/*.npy')

dataset = []

buckets = 100

a = [(bucket*buckets,bucket*buckets+buckets) for bucket in range(10)]

def find(d):
    for index, item in enumerate(a):
        if item[0] < d and item[1] > d:
                return index
    return -1

for file_ in files:
    X_positions = np.load(file_)

    X = X_positions

    X = X[::10]

    # Create Training dataset from this sequence
    #print(X.shape[0])-> 1002
    for frame_num in range(X.shape[0]):
        f = find(frame_num)
        if f != -1:
            dataset.append((f,X[frame_num,:,:]))

new_dataset = []
for batch in range(int(len(dataset)/batch_size)):

    batched = []
    for item in dataset[batch*batch_size:batch*batch_size + batch_size]:

        bnum, data = item
        h = torch.tensor([bnum])
        h = F.one_hot(h,num_classes=10)
        new_data = np.concatenate([h, data.reshape(1,120)],-1).reshape(130)
        batched.append(new_data)

    batched = np.stack(batched)
    new_dataset.append(batched)

dataset = new_dataset

# Shuffle the dataset
import random
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

split = int(len(dataset)*0.8)
training_dataset = dataset[:split]
testing_dataset = dataset[split:]
random.shuffle(training_dataset)

# Dataset size
print(len(training_dataset))

##
# Autoencoder Class Definitions
##

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.mlp = nn.Linear(input_size,hidden_size)
        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        self.mlp3 = nn.Linear(hidden_size, output_size)
        self.mu = nn.Linear(output_size+label_size, output_size)
        self.var = nn.Linear(output_size+label_size,output_size)

    def forward(self,x,y):
        x = torch.cat([y,x],1)
        x = torch.sigmoid(self.mlp(x))
        x = torch.sigmoid(self.mlp2(x))
        x = torch.sigmoid(self.mlp3(x))
        mu = self.mu(torch.cat([y,x],1))
        var = self.var(torch.cat([y,x],1))
        return mu, var


class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.mlp = nn.Linear(input_size,hidden_size)
        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        self.mlp3 = nn.Linear(hidden_size, output_size)

    def forward(self,x):
        x = torch.sigmoid(self.mlp(x))
        x = torch.sigmoid(self.mlp2(x))
        x = self.mlp3(x)
        return x

class VAE(nn.Module):

    def __init__(self, input_size, label_size, hidden_size, output_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size+label_size, hidden_size, output_size)
        self.decoder = Decoder(output_size,hidden_size, input_size)

    def forward(self,x,y):
        mu, log_var = self.encoder(x,y)
        encoded = self.reparameterize(mu, log_var)
        pred_x = self.decoder(encoded)
        return pred_x, mu, log_var, encoded

    def reparameterize(self, mu, log_var):

        s = torch.exp(0.5*log_var)
        e = torch.randn_like(s).cuda()

        return mu + e*s

    def generate_step(self, label):
        t = torch.tensor([label])
        t = F.one_hot(t,num_classes=10)
        t = t.float().cuda()
        z = torch.normal(0,3,size=(1,30)).cuda()
        z = torch.cat([t, z],1)
        decoded = self.decoder(z)
        return decoded
        
##
# Variational Autoencoder Initalization
##
input_size = 120
label_size = 10
hidden_size = 60
output_size = 30
vae = VAE(input_size,label_size,hidden_size,output_size).cuda()

###
# Begin GAN Training
###
import torch.optim as optim
learning_rate = 1e-3

optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

##
# Loss Function
##
def loss_fn(recon_x, x, mean, log_var,y):
    MSE_loss = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (MSE_loss + KLD) / x.size(0)


max_epochs = 5
epoch_loss = []
for epoch in range(max_epochs):

    training_loss = []
    for data in training_dataset:

        x = torch.tensor(data[:,10:]).float().cuda()
        y = torch.tensor(data[:,:10]).float().cuda()
        pred_x, mu, log_var, encoded = vae(x,y)
        loss = loss_fn(pred_x, x, mu, log_var,y)
        optimizer.zero_grad()
        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    epoch_loss.append(np.mean(training_loss))
print('Done')

##
# Generation
##
vae.eval()
# Go through the reaction coordinate of the trajectory
max_generation_steps = 10
predictions = []
for t in range(max_generation_steps):
    gen_frame = vae.generate_step(t)
    predictions.append(gen_frame.view(40,3))

predictions = torch.stack(predictions)
predictions = predictions.cpu().detach().numpy()
# Save predictions into VMD format
frame_num = predictions.shape[0]

nAtoms = "40"
outName = "VAE.xyz"
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

##
# Plot Loss
##
plt.plot(range(len(epoch_loss)), epoch_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('loss.png')
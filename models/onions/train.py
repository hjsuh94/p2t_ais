import os 
import torch
import torchvision
from torch import nn 
from torch.autograd import Variable 
from torch.utils.data import DataLoader 
from torchvision import transforms 
from torchvision.utils import save_image 

import torch.multiprocessing
from tqdm import tqdm 
from torchsummary import summary 
from itertools import chain 
import numpy as np
from torch.utils.tensorboard import SummaryWriter 
import torch.nn.functional as F 

from dataloader import OnionDataset
import models 

import warnings


"""
0. Minor pre-settings for training.
"""
writer = SummaryWriter("runs/onion_net")
warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')

torch.cuda.empty_cache()
cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
1. Hyperparameters and training settings
"""
num_epochs = 1000
batch_size = 64
initial_lr = 1e-3

onion_dataset = OnionDataset("data.csv", "images", "/home/hsuh/Documents/p2t_ais/data/onions")

"""
2. Model Declaration
"""
z = 100 # latent vector dimension
a = 4   # action space dimension
reward = models.RewardMLP(z, a)
dynamics = models.DynamicsMLP(z, a)
compression = models.CompressionMLP(z,a)

model_lst = [reward, dynamics, compression]
param_lst = []

for model in model_lst:
    model.to(cuda_device) # send them to GPU.
    param_lst += list(model.parameters()) # Collect all parameters.    
    # TODO(terry-suh): add the option to load paramters from previous training.


"""
3. Optimizer
"""
optimizer = torch.optim.Adam(param_lst, lr=initial_lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20)

"""
4. Loss function
"""
loss = nn.MSELoss(reduction='sum')

"""
5. Training
"""
best_loss = 1e4
for epoch in range(num_epochs):

    # Set them to training mode.
    for model in model_lst:
        model.train()

    running_loss = 0.0

    for data in tqdm(dataloader):
        # Load batches of data
        image_i = Variable(data['image_i']).cuda(cuda_device)
        image_f = Variable(data['image_f']).cuda(cuda_device)
        u = Variable(data['u']).cuda(cuda_device).float()

        optimizer.zero_grad()

        # Compute z using compression.
        z_i = compression(image_i)
        z_f = compression(image_f)

        # Compute reward from action.
        rhat = reward(torch.cat((z_i,u), 1))

        # Compute estimate of z_f with dynamics.
        zhat_f = dynamics(torch.cat((z_i,u), 1))

        # Compute loss in state-space.
        z_loss = loss(zhat_f, z_f)

        # Compute loss in reward-space.
        r_loss = loss(rhat, rtrue)

        # Total loss, backprop, and SGD.
        weight = 0.9
        total_loss = weight * r_loss + (1 - weight) * z_loss
        running_loss += total_loss 

        total_loss.backward()
        optimizer.step()

    print('epoch[{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, train_loss.item()))
    for param_group in optimizer.param_groups:
        print(param_group['lr'])

    scheduler.step(running_loss)
    writer.add_scalar('training_loss', total_loss.item(), epoch)

    if (total_loss < best_loss):
        # TODO(terry-suh): save weights
        best_loss = train_loss
        print("Model Saved!")
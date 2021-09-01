import os 
import matplotlib.pyplot as plt
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

import warnings
from models import model_io
from models.single_integrator.model import CompressionConv, DeCompressionConv
from models.single_integrator.offline_dataloader import OfflineDataset

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.z = 10

        # Input image: 1 x 128 x 128
        self.compression_conv = nn.Sequential(
            nn.Conv2d(1, 4, 2, stride=2), # 4 x 64 x 64
            nn.ReLU(),
        )

        # after compressing & flattening, pass it through another layer of mlps
        # to get the desired dimension of z.
        self.compression_conv_mlp = nn.Sequential(
            nn.Linear(4 * 32 * 32, 100),
            nn.ReLU(),
            nn.Linear(100, self.z),
            nn.ReLU(),
        )

        self.decompression_conv = nn.Sequential(
            nn.ConvTranspose2d(4, 1, 2, stride=2),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Sigmoid()
        )

        self.decompression_conv_mlp = nn.Sequential(
            nn.Linear(self.z, 100),
            nn.Tanh(),
            nn.Linear(100, 4 * 32 * 32),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        b = x.shape[0]
        x = self.compression_conv(x)
        x = x.view(b, 4 * 32 * 32)
        x = self.compression_conv_mlp(x)
        x = self.decompression_conv_mlp(x)
        x = x.view(b, 4, 32, 32)
        x = self.decompression_conv(x)
        return x.squeeze(1)

class AutoEncoderCarrot(nn.Module):
    def __init__(self):
        super(AutoEncoderCarrot, self).__init__()

        self.z = 100

        # after compressing & flattening, pass it through another layer of mlps
        # to get the desired dimension of z.
        self.compression_conv_mlp = nn.Sequential(
            nn.Linear(32 * 32, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, self.z),
            nn.Tanh()
        )

        self.decompression_conv_mlp = nn.Sequential(
            nn.Linear(self.z, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 32 * 32),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        b = x.shape[0]
        x = x.view(b,  32 * 32)
        x = self.compression_conv_mlp(x)
        x = self.decompression_conv_mlp(x)
        x = x.view(b, 1, 32, 32)
        return x.squeeze()



"""
1. Load dataloader 
"""

offline_dataset = OfflineDataset(
    "data.csv", "images",
    "/home/hsuh/Documents/p2t_ais/data/carrot"
)
offline_dataloader = DataLoader(offline_dataset, batch_size=256,
    shuffle=True, num_workers=24)

'''
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32,32)),
    torchvision.transforms.ToTensor()]
)
train_dataset = torchvision.datasets.MNIST(
    root="torch_dataset", train=True, transform=transform, download=True
)
offline_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=256, shuffle=True, num_workers=24)
'''
"""
2. Define Models
"""

autoencoder = AutoEncoderCarrot().cuda().train()
optimizer = torch.optim.Adam(autoencoder.parameters(),
    lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
loss = nn.BCELoss()

"""
3. Train
"""

for epoch in range(100):
    running_loss = 0.0

    for data in tqdm(offline_dataloader):
        optimizer.zero_grad()

        image_i = Variable(data["image_i"]).cuda()

        image_i_copy = image_i.clone().detach()

        total_loss = loss(autoencoder(image_i), image_i_copy)
        total_loss.backward()

        optimizer.step()

        running_loss += total_loss.item() 

    lr_scheduler.step(running_loss)

    print('epoch[{}/{}], loss: {:4f}'.format(epoch+1, 100, running_loss))


plt.figure()
plt.imshow(image_i_copy[0,:,:].squeeze(0).cpu().numpy(), cmap='gray', vmin=0.0, vmax=1.0)
plt.show()

plt.figure()
plt.imshow(autoencoder(image_i_copy[0,:,:].unsqueeze(0)).detach().cpu().numpy(), cmap='gray', vmin=0.0, vmax=1.0)    
plt.show()



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
writer = SummaryWriter("runs/onion_net_lindyn")
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
dataloader = DataLoader(onion_dataset, batch_size=batch_size, \
    shuffle=True, num_workers=24)

"""
2. Model Declaration
"""
z = 100 # latent vector dimension
a = 4   # action space dimension
reward = models.RewardMLP(z, a)
#dynamics = models.DynamicsMLP(z, a)
dynamics = models.DynamicsLinear(z, a)
compression = models.CompressionMLP(z,a)
#compression = models.CompressionLinear(z,a)

model_lst = [reward, dynamics, compression]
param_lst = []

model_dir = "/home/hsuh/Documents/p2t_ais/models/onions/weights"
model_name_lst = [
    "reward_mlp_lindyn.pth",
    "dynamics_mlp_lindyn.pth",
    "compression_lindyn.pth"
]

for i in range(len(model_lst)):
    model = model_lst[i]
    model_name = model_name_lst[i]

    model.to(cuda_device) # send model to GPU
    param_lst += list(model.parameters()) # collect all parameters
    # NOTE(terry-suh): load parameters from previous training based on model_name_lst
    try:
        model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
    except FileNotFoundError:
        print("Model " + model_name + " not found.")
        pass
    model.eval()


"""
3. Optimizer
"""
optimizer = torch.optim.Adam(param_lst, lr=initial_lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20)

"""
4. Loss function. Includes Lyapunov function implementation.
"""
loss = nn.MSELoss(reduction='sum')

def lyapunov_measure():
    """
    Return lyapunov measure by creating a weighted matrix.
    """
    pixel_radius = 7
    measure = np.zeros((32, 32))
    for i in range(32):
        for j in range(32):
            radius = np.linalg.norm(np.array([i - 15.5, j - 15.5]), ord=2)
            measure[i,j] = np.maximum(radius - pixel_radius, 0)
    return measure

def lyapunov(image):
    """
    Apply the lyapunov measure to the image. Expects (B x 1 x 32 x 32), output B vector.
    """
    V_measure = torch.Tensor(lyapunov_measure()).to(cuda_device)
    return torch.sum(torch.mul(image, V_measure), [2,3])


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
        rtrue = lyapunov(image_f) - lyapunov(image_i)
        r_loss = loss(rhat, rtrue)

        # Total loss, backprop, and SGD.
        weight = 0.9
        total_loss = weight * r_loss + (1 - weight) * z_loss
        running_loss += total_loss 

        total_loss.backward()
        optimizer.step()

    print('epoch[{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, total_loss.item()))
    for param_group in optimizer.param_groups:
        print(param_group['lr'])

    scheduler.step(running_loss)
    writer.add_scalar('training_loss', total_loss.item(), epoch)

    if (total_loss < best_loss):
        # TODO(terry-suh): save weights
        for i in range(len(model_lst)):
            model = model_lst[i]
            model_name = model_name_lst[i]
            torch.save(model.state_dict(), os.path.join(model_dir, model_name))

        best_loss = total_loss
        print("Model Saved!")
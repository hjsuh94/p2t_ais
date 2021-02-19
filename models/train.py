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

import warnings
from models import model_io

def train(model_dict, lmbda, dataloader, optimizer, num_epochs, lr_scheduler, 
    torch_writer, model_save_dict, model_save_dir, save='best', reduction='sum', cuda_device="cuda:0"):
    """
    """
    #NOTE(terry-suh): the reduction is built into the function
    loss = nn.MSELoss(reduction)

    best_loss = 1e5
    compression = model_dict["compression"]
    reward = model_dict["reward"]
    dynamics = model_dict["dynamics"]

    for epoch in range(num_epochs):
        for key in ["dynamics", "compression", "reward"]:
            model = model_dict[key]
            model.train()

        running_loss = 0.0

        for data in tqdm(dataloader):
            image_i = Variable(data['image_i']).cuda(cuda_device)
            image_f = Variable(data['image_f']).cuda(cuda_device)
            u = Variable(data['u']).cuda(cuda_device).float()
            rtrue = Variable(data['r']).cuda(cuda_device).float()

            optimizer.zero_grad()

            z_i = compression(image_i)
            z_f = compression(image_f)

            rhat = reward(torch.cat((z_i, u), 1))
            zhat_f = dynamics(torch.cat((z_i, u), 1))
            z_loss = loss(zhat_f, z_f)
            r_loss = loss(rhat, rtrue)

            # Below is only for linear weights. comment out for usual.
            #compression.R.weights

            total_loss = lmbda * r_loss + (1 - lmbda) * z_loss 
            running_loss += total_loss 

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                compression.L.weight[:] = compression.L.weight.clamp(0, 1)
                dynamics.A.weight[:] = dynamics.A.weight.clamp(0, 1)                
                

        print('epoch[{}/{}], loss: {:4f}'.format(epoch+1, num_epochs, total_loss.item()))
        torch_writer.add_scalar('training_loss', total_loss.item(), epoch)

        lr_scheduler.step()
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        if (save == "best"):
            if (total_loss < best_loss):
                model_io.save_models(model_dict, model_save_dict, model_save_dir)
                best_loss = total_loss 
                print("model saved!")

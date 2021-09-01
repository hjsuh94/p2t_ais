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

import importlib

import warnings
from models import model_io

def train(model_dict, dataloader, config, torch_writer):

    if "decompression" in model_dict:
        raise ValueError("Not allowed to train an interpreter because there is already a decoder.")

    # 1. Declare interpreter Model
    model_module = importlib.import_module("models." + config["env"] + ".model")
    num_z = config["model"]["z"]
    num_a = config["model"]["a"]
    decompression = getattr(model_module,
        config["interpret"]["decompression"])(num_z, num_a)
    decompression = decompression.cuda()

    # 2. Declare optimizer
    optimizer = torch.optim.Adam(decompression.parameters(), lr=config["offline_train"]["initial_lr"])
    lr_scheduler = getattr(torch.optim.lr_scheduler, config["offline_train"]["lr_scheduler"])

    if (config["offline_train"]["lr_scheduler"] == "StepLR"):
        scheduler = lr_scheduler(optimizer, step_size=config["offline_train"]["lr_step_size"])
    else:
        scheduler = lr_scheduler(optimizer, patience=20)

    # 3. 
    loss = nn.MSELoss(reduction='mean')

    best_loss = np.inf
    compression = model_dict["compression"]
    compression.eval()

    num_epochs = config["offline_train"]["num_epochs"]

    for epoch in range(num_epochs):
        decompression.train()

        # 4. Train the decoder.

        running_loss = 0.0

        for data in tqdm(dataloader):

            image_i = Variable(data["image_i"]).cuda()

            optimizer.zero_grad()

            z_i = compression(image_i)
            image_hat = decompression(z_i)

            total_loss = loss(image_hat, image_i)

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        print('epoch[{}/{}], loss: {:4f}'.format(epoch+1, num_epochs, running_loss))
        torch_writer.add_scalar('training_loss', running_loss, epoch)

        # 5. Update and look at the images.
        decompression.eval()

        b = image_i.shape[0]
        b_index = np.random.randint(b)
        image_i_sample = image_i[b_index,:,:].unsqueeze(0)
        image_i_reconstruction = decompression(compression(image_i_sample))

        torch_writer.add_image("original", image_i_sample.squeeze(0), epoch)
        torch_writer.add_image("reconstructed", image_i_reconstruction.squeeze(0), epoch)

        scheduler.step()
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        if (running_loss < best_loss):
            torch.save(decompression.state_dict(), os.path.join(
                config["save_model"]["model_dir"], "interpreter2.pth"))
            best_loss = running_loss 
            print("model saved!")


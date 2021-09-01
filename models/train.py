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
from policies.policy import rollout_batch

# TODO(terry-suh): Move this to carrot specific folder.
from models.carrot.data_augmentation import rotate_data

def train(model_dict, train_config, config, dataloader, optimizer, lr_scheduler, 
    torch_writer, save='best', reduction='sum', cuda_device="cuda:0", write_results=True):
    """
    """

    lmbda = train_config["lmbda"]
    num_epochs = train_config["num_epochs"]
    loss_mode = train_config["mode"]
    model_save_dict = config["save_model"]
    model_save_dir = os.path.join(config["path"], config["save_model"]["model_dir"])
    num_models = config["model"]["num_models"]
    
    #NOTE(terry-suh): the reduction is built into the function
    loss = nn.MSELoss(reduction='mean')

    best_loss = np.inf
    
    key_lst = ["dynamics", "compression", "reward"]
    compression = model_dict["compression"]
    reward = model_dict["reward"]
    dynamics = model_dict["dynamics"]

    if "decompression" in model_dict:
        key_lst.append("decompression")
        decompression = model_dict["decompression"]

    for epoch in range(num_epochs):
        for key in key_lst:
            model = model_dict[key]
            model.train()

        running_loss = 0.0
        running_loss_rz = 0.0

        for data in tqdm(dataloader):

            optimizer.zero_grad()

            if (loss_mode == "equation"):

                image_i = Variable(data['image_i']).cuda(cuda_device)
                image_f = Variable(data['image_f']).cuda(cuda_device)
                u = Variable(data['u']).cuda(cuda_device).float()
                rtrue = Variable(data['r']).cuda(cuda_device).float()

                # Data augmentation.
                result = rotate_data(image_i, image_f, u, device=cuda_device)
                image_i = result["image_i"]
                image_f = result["image_f"]
                u = result["u"]

                zu_concat_dim = 1

                # If we're training an ensemble, repeat tensors along ensemble dimension. 
                if (num_models > 1):
                    image_i = torch.stack(num_models * [image_i])
                    image_f = torch.stack(num_models * [image_f])
                    u = torch.stack(num_models * [u])
                    rtrue = torch.stack(num_models * [rtrue])
                    zu_concat_dim = 2

                z_i = compression(image_i)
                z_f = compression(image_f)

                rhat = reward(torch.cat((z_i, u), zu_concat_dim))
                zhat_f = dynamics(torch.cat((z_i, u), zu_concat_dim))

                z_loss = loss(zhat_f, z_f)
                r_loss = loss(rhat, rtrue)

                total_loss = lmbda * r_loss + (1 - lmbda) * z_loss 
                running_loss_rz += total_loss

                if "decompression" in model_dict:
                    image_hat = decompression(z_f)
                    total_loss += 1.0 * loss(image_f, image_hat)

            if (loss_mode == "simulation"):

                image_i = Variable(data['image_i']).cuda(cuda_device)
                input_traj = Variable(data['input_traj']).cuda(cuda_device).float()
                reward_traj = Variable(data['reward_traj']).cuda(cuda_device).float()

                result = rollout_batch(image_i, model_dict, config["dataset"]["horizon"], input_traj)
                rhat_traj = result["r_mat"]

                for t in range(reward_traj.shape[1]):
                    reward_traj[:,t] = np.power(0.8, t) * reward_traj[:,t]
                    rhat_traj[:,t] = np.power(0.8, t) * rhat_traj[:,t]                    

                total_loss = loss(reward_traj, rhat_traj)

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        print('epoch[{}/{}], loss: {:4f}'.format(epoch+1, num_epochs, running_loss))

        if (write_results):
            torch_writer.add_scalar('training_loss', running_loss, epoch)
            torch_writer.add_scalar('rz_loss', running_loss_rz, epoch)

        for key in key_lst:
            model = model_dict[key]
            model.eval()

        if "decompression" in model_dict:
            b = image_i.shape[0]
            b_index = np.random.randint(b)
            image_i_sample = image_i[b_index,:,:].unsqueeze(0)
            image_i_reconstruction = decompression(compression(image_i_sample))

            torch_writer.add_image("original", image_i_sample.squeeze(0), epoch)
            torch_writer.add_image("reconstructed", image_i_reconstruction.squeeze(1), epoch)

        if num_models > 1:
            b = image_i.shape[1] # Pick along batch dimension
            b_index = np.random.randint(b)
            # Need this to be E x 1 x 
            image_i_sample = image_i[:,b_index,:,:,:].unsqueeze(1)
            u_sample = u[:,b_index,:].unsqueeze(1)
            r_sample = rtrue[:,b_index,:].unsqueeze(1)

            z_i = compression(image_i_sample)
            zhat_f = dynamics(torch.cat((z_i, u_sample), 2))
            rhat_f = reward(torch.cat((zhat_f, u_sample), 2))

            rhat_f_np = rhat_f.detach().cpu().numpy()
            print(rhat_f_np)
            



        #lr_scheduler.step(running_loss)
        lr_scheduler.step()
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        if (save == "best"):
            if (running_loss < best_loss):
                model_io.save_models(model_dict, model_save_dict, model_save_dir)
                best_loss = running_loss 
                print("model saved!")

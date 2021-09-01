import torch 
import torchvision.transforms.functional as FT
from models.carrot.offline_dataloader import OfflineDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

def visualize_instance(image_i, image_f, u, image_ip, image_fp, up):
    fig, ax = plt.subplots(2, 2)
    ax[0,0].imshow(image_i, cmap='gray')
    ax[0,1].imshow(image_f, cmap='gray')

    u_pixel = 32.0 * (u + 0.5)
    ax[0,0].plot(u_pixel[0], 32 - u_pixel[1], 'ro')
    ax[0,0].plot(u_pixel[2], 32 - u_pixel[3], 'go')    
    ax[0,0].plot([u_pixel[0], u_pixel[2]], 32 - np.array([u_pixel[1], u_pixel[3]]), 'g-')

    ax[1,0].imshow(image_ip, cmap='gray')
    ax[1,1].imshow(image_fp, cmap='gray')

    up_pixel = 32.0 * (up + 0.5)
    ax[1,0].plot(up_pixel[0], 32 - up_pixel[1], 'ro')
    ax[1,0].plot(up_pixel[2], 32 - up_pixel[3], 'go')    
    ax[1,0].plot([up_pixel[0], up_pixel[2]], 32 - np.array([up_pixel[1], up_pixel[3]]), 'g-')

    plt.show()

def rotate_u(u, degree, device="cuda:0"):
    # 1. Convert u to pixel coordinates.
    u_p = 32.0 * (torch.stack([u[:,0], -u[:,1], u[:,2], -u[:,3]], dim=1))
    
    # 2. Cretae rotation matrix. 
    rad = torch.deg2rad(torch.Tensor([degree]))
    R = torch.Tensor([
        [torch.cos(rad), torch.sin(rad)],
        [-torch.sin(rad), torch.cos(rad)]
    ]).cuda(device)

    # 3. Apply rotation matrix. 
    u_fp = torch.cat(
        [
            (torch.mm(R, u_p[:,0:2].T)).T, 
            (torch.mm(R, u_p[:,2:4].T)).T, 
        ],
        dim=1)

    # 4. Convert back to original coordinates  
    u_f = (torch.stack([u_fp[:,0], -u_fp[:,1], u_fp[:,2], -u_fp[:,3]], dim=1) / 32.0)

    return u_f

def rotate_data(image_i, image_f, u, device="cuda:0"):
    """
    Randomly rotate data for data augmentation. The augmentation is reward-invariant.
    """
    degree = 360.0 * np.random.rand()

    image_ip = FT.rotate(image_i, degree)
    image_fp = FT.rotate(image_f, degree)
    u_p = rotate_u(u, degree, device=device)

    result = {"image_i": image_ip, "image_f": image_fp, "u": u_p}
    return result

# Below is test code.
""" 
offline_dataset = OfflineDataset("data.csv", "images", "/home/hsuh/Documents/p2t_ais/data/carrot")
dataloader = DataLoader(offline_dataset, batch_size=256, shuffle=True, num_workers=4)

for data in tqdm(dataloader):
    image_i = data["image_i"]
    image_f = data["image_f"]
    u = data["u"]
    r = data["r"]

    image_i_sample = image_i[0,0,:,:].detach().cpu().numpy()
    image_f_sample = image_f[0,0,:,:].detach().cpu().numpy()
    u_sample = u[0,:].detach().cpu().numpy()

    degree = 360.0 * np.random.rand()

    image_ip = FT.rotate(image_i, degree)
    image_fp = FT.rotate(image_f, degree)
    u_p = rotate_u(u, degree)

    image_i_samplep = image_ip[0,0,:,:].detach().cpu().numpy()
    image_f_samplep = image_fp[0,0,:,:].detach().cpu().numpy()
    u_samplep = u_p[0,:].detach().cpu().numpy()    

    print("rotated by " + str(degree))

    visualize_instance(image_i_sample, image_f_sample, u_sample, image_i_samplep, image_f_samplep, u_samplep)
"""

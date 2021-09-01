import os
import torch 
import csv
from skimage import io, transform 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd 
from PIL import Image 

import warnings 
warnings.filterwarnings("ignore")

class ReplayDataset(Dataset):
    """
    Torch dataset class for accessing onion dataset
    """
    def __init__(self, SARSlist):
        self.SARSlist = SARSlist


    def __len__(self):
        return len(self.SARSlist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        SARS = self.SARSlist[idx]

        # Each image will be downsampled to 32 by 32.
        image_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((32,32)),
            transforms.ToTensor(),
        ])

        image_i = image_transform(Image.fromarray(SARS[0]))
        image_f = image_transform(Image.fromarray(SARS[3]))
        u = SARS[1]
        r = SARS[2]
        
        sample = {'image_i': image_i, 'image_f': image_f, 'u': u, 'r': r}

        return sample

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

class OnionDataset(Dataset):
    """
    Torch dataset class for accessing onion dataset
    """
    def __init__(self, csv_file, data_dir, root_dir):

        self.csv_filename = os.path.join(root_dir, csv_file)
        self.data_dir = os.path.join(root_dir, data_dir)
        self.csv_file = pd.read_csv(self.csv_filename)

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imagename_i = os.path.join(self.data_dir, \
            self.csv_file.iloc[idx, 0])
        imagename_f = os.path.join(self.data_dir, \
            self.csv_file.iloc[idx, 5])

        # Each image will be downsampled to 32 by 32.
        image_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((32,32)),
            transforms.ToTensor(),
        ])

        image_i = image_transform(Image.open(imagename_i, 'r'))
        image_f = image_transform(Image.open(imagename_f, 'r'))

        u = self.csv_file.iloc[idx,1:5].to_numpy(dtype=np.double)

        sample = {'image_i': image_i, 'image_f': image_f, 'u': u}

        return sample

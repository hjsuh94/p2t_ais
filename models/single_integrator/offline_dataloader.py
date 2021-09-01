import os
import torch 
import csv
from skimage import io, transform 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd 
import cv2
import time 
import warnings 
import matplotlib.pyplot as plt
from PIL import Image
warnings.filterwarnings("ignore")

class OfflineDataset(Dataset):
    """
    Torch dataset class for accessing onion dataset
    """
    def __init__(self, csv_file, data_dir, root_dir):

        self.csv_filename = os.path.join(root_dir, csv_file)
        self.csv_file = pd.read_csv(self.csv_filename)        
        self.data_dir = os.path.join(root_dir, data_dir)

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        episode = self.csv_file.iloc[idx, 0]
        episode_dir = os.path.join(self.data_dir, "{:04d}".format(episode))


        imagename_i = os.path.join(episode_dir,
            self.csv_file.iloc[idx, 1])
        imagename_f = os.path.join(episode_dir,
            self.csv_file.iloc[idx, 2])

        r = np.array([self.csv_file.iloc[idx, 3]])
        u = self.csv_file.iloc[idx,4:6].to_numpy(np.float32)

        image_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        image_i = image_transform(Image.open(imagename_i, 'r'))
        image_f = image_transform(Image.open(imagename_f, 'r'))


        sample = {'image_i': image_i, 'image_f': image_f, 'u': u, 'r': r}

        return sample

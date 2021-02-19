import os
import torch 
import csv
from skimage import io, transform 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd 
import cv2

import warnings 
warnings.filterwarnings("ignore")

class OfflineDataset(Dataset):
    """
    Torch dataset class for accessing onion dataset
    """
    def __init__(self, csv_file, data_dir, root_dir):

        self.csv_filename = os.path.join(root_dir, csv_file)
        self.csv_file = pd.read_csv(self.csv_filename)        
        self.data_dir = os.path.join(root_dir, data_dir)

        print(len(self.csv_file))

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

        r = self.csv_file.iloc[idx, 3]
        u = self.csv_file.iloc[idx,4:8].to_numpy(np.float32)
        
        image_i = torch.Tensor(cv2.imread(imagename_i, cv2.IMREAD_GRAYSCALE).astype(np.float)) / 255.
        image_f = torch.Tensor(cv2.imread(imagename_f, cv2.IMREAD_GRAYSCALE).astype(np.float)) / 255.

        sample = {'image_i': image_i, 'image_f': image_f, 'u': u, 'r': r}

        return sample

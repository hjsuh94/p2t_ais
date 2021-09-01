import os
import torch 
import csv
from skimage import io, transform 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd 
import cv2
from PIL import Image

import warnings 
warnings.filterwarnings("ignore")

class OfflineDataset(Dataset):
    """
    Torch dataset class for accessing onion dataset
    """
    def __init__(self, csv_file, data_dir, root_dir, length="all"):

        self.csv_filename = os.path.join(root_dir, csv_file)
        self.csv_file = pd.read_csv(self.csv_filename, header=None)        
        self.data_dir = os.path.join(root_dir, data_dir)
        self.length = length

    def __len__(self):
        if (self.length == "all"):
            print(len(self.csv_file))
            return len(self.csv_file)
        else:
            return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        episode = self.csv_file.iloc[idx, 0]
        episode_dir = os.path.join(self.data_dir, "{:04d}".format(episode))


        imagename_i = os.path.join(episode_dir,
            self.csv_file.iloc[idx, 1])
        imagename_f = os.path.join(episode_dir,
            self.csv_file.iloc[idx, 2])

        # Make sure this doesn't become a scalar so we have B x 1 dim for reward, NOT B.
        r = np.array([self.csv_file.iloc[idx, 3]])
        u = self.csv_file.iloc[idx,4:8].to_numpy(np.float32)

        image_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        image_i = image_transform(Image.open(imagename_i, 'r'))
        image_f = image_transform(Image.open(imagename_f, 'r'))

        sample = {'image_i': image_i, 'image_f': image_f, 'u': u, 'r': r}

        return sample

class SimulationDataset(Dataset):
    """
    Torhc dataset class for doing simulations.
    """

    def __init__(self, npy_filename, data_dir, root_dir):
        self.npy_filename = os.path.join(root_dir, npy_filename)
        self.data_lst = np.load(self.npy_filename, allow_pickle=True)
        self.data_dir = os.path.join(root_dir, data_dir)

    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data_lst[idx]
        episode = int(data[0])
        episode_dir = os.path.join(self.data_dir, "{:04d}".format(episode))
        imagename_i = os.path.join(episode_dir, data[1])

        image_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        image_i = image_transform(Image.open(imagename_i, 'r'))

        reward_traj = np.array(data[2])
        input_traj = np.array(data[3])

        sample = {"image_i": image_i, "reward_traj": reward_traj, "input_traj": input_traj}

        return sample

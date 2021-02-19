import torch
from torchvision import transforms
import numpy as np 
from PIL import Image 
import cv2

import matplotlib.pyplot as plt


def image_to_tensor(image_cv2):
    """
    Convert cv2 image of HxW into a torch representation of 1 x 1 x H x W. 
    """
    image_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    # 1. cv2 to PIL 
    image = Image.fromarray(image_cv2)
    # 2. PIL to tensor.
    image = image_transform(image)
    # 3. tensor to batch tensor 
    image = torch.unsqueeze(image, 0)
    return image 

def plot_arrow(u, value, value_range, colormap='jet', width=2):
    """
    Plot arrow that corresponds to action. 
    Color arrow by the quantity given in value, under the assumption that
    value_range = [v_min, v_max].
    """
    # flip y-axis in u for plotting.
    u = np.array([u[0], -u[1], u[2], -u[3]])
    # convert u from [-0.5, 0.5] to [0, 500]
    u_scaled = 500.0 * (0.5 + u)

    # compute value and compute associated color.
    cmap = plt.get_cmap(colormap)
    scale = float(value_range[1] - value_range[0])
    color = cmap(1 - float(value - value_range[0]) / scale)

    # draw arrow. 
    plt.arrow(u_scaled[0],
              u_scaled[1],
              u_scaled[2] - u_scaled[0],
              u_scaled[3] - u_scaled[1],
              width = width,
              fc = color,
              ec = color)
    return None



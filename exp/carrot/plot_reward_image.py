import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
import csv, os
import pandas as pd 

import torch
from sim.onions.onion_sim import OnionSim
from exp.onions.utils import plot_arrow, image_to_tensor
import models.onions.models as models

"""
0. Set up simulation. Test a random action.
"""
sim = OnionSim()
sim.refresh()
#sim.update([0.4, 0.4, 0.0, 0.0])
#sim.update([0.3, 0.4, -0.1, 0.0])
#sim.update([0.5, 0.4, 0.1, 0.0])

"""
1. Set up models.
"""
z = 100
a = 4 
reward = models.RewardMLP(z, a)
dynamics = models.DynamicsMLP(z, a)
compression = models.CompressionMLP(z, a)

model_lst = [reward, dynamics, compression]
model_dir = "/home/hsuh/Documents/p2t_ais/models/onions/weights"
model_name_lst = [
    "reward_mlp.pth",
    "dynamics_mlp.pth",
    "compression_mlp.pth"
]

cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for i in range(len(model_lst)):
    model = model_lst[i]
    model_name = model_name_lst[i]

    model.to(cuda_device)
    try:
        model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
    except FileNotFoundError:
        print("Model " + model_name + " not found.")
        pass 
    model.eval()

"""
2. Plot images and arrows....
"""

image_np = sim.get_current_image()
image_tensor = image_to_tensor(image_np).cuda(cuda_device)
z = compression(image_tensor)
plt.figure()
plt.imshow(image_np)

num_arrows = 100

rhat_lst = []
for i in range(500):
    u = -0.4 + 0.8 * np.random.rand(4)
    u_tensor = torch.unsqueeze(torch.Tensor(u).cuda(cuda_device), 0)
    rhat = reward(torch.cat((z, u_tensor), 1)).cpu().detach().numpy()
    if (rhat <= 0):
        plot_arrow(u, -rhat, [0, 30], colormap="jet")

#print(np.max(np.array(rhat_lst)))
#print(np.min(np.array(rhat_lst)))

plt.colorbar(
    matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0, vmax=30, clip=False),
        cmap='jet'
    ))
plt.show()





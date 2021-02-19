import numpy as np
import cv2 
import csv 
import os, time
from sim.onions.onion_sim import OnionSim
from models.onions.rewards import lyapunov, lyapunov_measure
import models.onions.models as models
from exp.onions.utils import image_to_tensor
from PIL import Image

import torch 
from torchvision import transforms, utils

"""
The purpose of this script is to evaluate reward predicting capabilities of the
models that are trained using the AIS framework. The evaluation will proceed 
with these two schemes.

1. Equation error.
2. Simulation error.

The data files are 
"""

"""
0. Setup writer.
TODO(terry-suh): some of these belong in something like utils.
"""

f = open("/home/hsuh/Documents/p2t_ais/exp/onions/prediction.csv", "w")
writer = csv.writer(f, delimiter=',', quotechar='|')

"""
1. Set up simulation.
"""
sim = OnionSim()
sim.refresh()

"""
2. Setup models
"""
z = 100
a = 4
reward = models.RewardMLP(z, a)
dynamics = models.DynamicsMLP(z, a)
compression = models.CompressionMLP(z, a)

model_lst = [reward, dynamics, compression]
model_dir = "/home/hsuh/Documents/p2t_ais/models/onions/weights"
model_name_lst = [
    "reward_mlp_online2.pth",
    "dynamics_mlp_online2.pth",
    "compression_mlp_online2.pth"
]

cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for i in range(len(model_lst)):
    model = model_lst[i]
    model_name = model_name_lst[i]

    model.to(cuda_device)
    try:
        model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
    except FileNotFoundError:
        print("Model " + model_name + "not found.")
        pass
    model.eval()

"""
3. Evaluate trajectories.
"""
sim.refresh()

# 3.1 Sample a random input trajectory open-loop.
traj_length = 8
u_lst = -0.4 + 0.8 * np.random.rand(traj_length, 4)

# Sub: create u_lst that makes sense?


u_lst = np.array([
    [0.0, 0.4, 0.0, 0.0],
    [0.0, -0.4, 0.0, 0.0],
    [0.4, 0.0, 0.0, 0.0],
    [-0.4, 0.0, 0.0, 0.0],
    [0.4, 0.4, 0.0, 0.0],
    [0.4, -0.4, 0.0, 0.0],
    [-0.4, 0.4, 0.0, 0.0],
    [-0.4, -0.4, 0.0, 0.0],
    ])


# 3.2 Predict the reward of this trajectory.
tensor_image = image_to_tensor(sim.get_current_image()).cuda()
V_now = lyapunov(tensor_image, device=cuda_device)
print("V_now: " + str(V_now))

V_change = 0.0
z = compression(tensor_image)

for i in range(traj_length):
    u = u_lst[i,:]
    u_tensor = torch.unsqueeze(torch.Tensor(u).cuda(cuda_device), 0)
    r_hat = reward(torch.cat((z, u_tensor), 1))
    z = dynamics(torch.cat((z, u_tensor), 1)) # compute next z (in-place)
    V_change += r_hat

V_lst = [float(V_now.cpu().numpy())]
V_hat_lst = [float(V_now.cpu().numpy())]

# 3.3 Run this open loop trajectory in sim.
sim.refresh()
for i in range(traj_length):
    tensor_image = image_to_tensor(sim.get_current_image()).cuda() # 1 x 1 x 32 x 32 
    V_now = lyapunov(tensor_image, device=cuda_device)
    # 3.2.1 Predict  
    u = u_lst[i,:]
    u_tensor = torch.unsqueeze(torch.Tensor(u).cuda(cuda_device), 0)
    z_i = compression(tensor_image)
    z_hat = dynamics(torch.cat((z_i, u_tensor), 1))
    rhat = reward(torch.cat((z_i, u_tensor), 1))

    V_hat_lst.append(float((V_hat_lst[-1] + rhat).cpu().detach().numpy()))
    #print(rhat)

    # 3.2.2. Rollout sim for actual behavior.
    sim.update(u)
    V_next = lyapunov(image_to_tensor(sim.get_current_image()).cuda(), device=cuda_device)
    V_lst.append(float(V_next.cpu().numpy()))
    #print(V_next - V_now)    
    #time.sleep(1.0)
    #print("executed")

writer.writerow(np.array(V_lst))
writer.writerow(np.array(V_hat_lst))









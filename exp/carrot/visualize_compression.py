import models.carrot.model as model
import torch
import os 
import numpy as np

import matplotlib.pyplot as plt

z = 10
a = 4
compression = model.CompressionLinearNoBias(z, a)
reward = model.RewardLinearNoAction(z, a)
dynamics = model.DynamicsLinearNoAction(z, a)

model_dir = "/home/hsuh/Documents/p2t_ais/models/carrot/weights"
compression_model_name = "no_action_linear/compression.pth"
reward_model_name = "no_action_linear/reward.pth"
dynamics_model_name = "no_action_linear/dynamics.pth"

compression.eval()
reward.eval()
dynamics.eval()

compression.load_state_dict(torch.load(os.path.join(model_dir, compression_model_name)))
reward.load_state_dict(torch.load(os.path.join(model_dir, reward_model_name)))
dynamics.load_state_dict(torch.load(os.path.join(model_dir, dynamics_model_name)))

L = compression.L.weight.detach().numpy()
A = dynamics.A.weight.detach().numpy()
R = reward.R.weight.detach().numpy()

"""
1. Visualize compression
"""


kernel_mat_sum = np.zeros((32,32))
reward_mat_sum = np.zeros((32,32))
for i in range(A.shape[0]):
    row = L[i,:]

    kernel_mat = torch.Tensor(row).view(32,32).numpy()
    reward_mat_sum += kernel_mat * R[:,i]
    kernel_mat_sum += kernel_mat

    plt.figure()
    plt.imshow(kernel_mat, cmap='gray')#, vmin=0, vmax=1)
    plt.colorbar()
    plt.savefig("/home/hsuh/Documents/p2t_ais/exp/carrot/images/kernels/" + "{:03d}".format(i) + ".png")
    plt.close()

plt.figure()
plt.imshow(kernel_mat_sum, cmap='gray')
plt.colorbar()
plt.savefig("/home/hsuh/Documents/p2t_ais/exp/carrot/images/kernels/" + "sum.png")
plt.close()

"""
2. Visualize dynamics
"""
plt.figure() 
plt.imshow(A, cmap='gray')
plt.colorbar()
plt.savefig("/home/hsuh/Documents/p2t_ais/exp/carrot/images/kernels/dynamics.png")
plt.close()

"""
3. Visualize rewards
"""
plt.figure()
plt.imshow(reward_mat_sum, cmap='gray')
plt.colorbar()
plt.savefig("/home/hsuh/Documents/p2t_ais/exp/carrot/images/kernels/reward.png")
plt.close()



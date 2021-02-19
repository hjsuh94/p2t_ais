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
We will test the AIS model with feedback in order to do a simple test, which
starts with a simple brute-force greedy controller that looks ahead one step 
into the future and chooses the steepest-descent along Lyapunov. 
"""

"""
0. Setup writer
"""
f = open("/home/hsuh/Documents/p2t_ais/exp/onions/closed_loop.csv", "w")
writer = csv.writer(f, delimiter=',', quotechar='|')

image_dir = "/home/hsuh/Documents/p2t_ais/exp/onions/images/"

"""
1. Setup simulation.
"""
sim = OnionSim()
sim.change_onion_num(100)
#sim.RENDER_EVERY_TIMESTEP = True
sim.refresh()

"""
2. Setup models.
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
3. Evaluate steepest descent controller.
"""

traj_length = 32
candidate_num = 3000

V_lst = []
Vhat_lst = []

for i in range(traj_length):
    # 0. Save current image. 
    filename = "steepest_descent/" + "{:03d}".format(i) + ".png"
    cv2.imwrite(os.path.join(image_dir, filename), sim.get_current_image())

    # 1. Feedback on current tensor image.
    tensor_image = image_to_tensor(sim.get_current_image()).cuda() # 1 x 1 x 32 x 32
    tensor_image_batch = tensor_image.repeat(candidate_num, 1, 1, 1) # C x 1 x 32 x 32
    V = lyapunov(tensor_image, device=cuda_device).cpu().detach().numpy()
    V_lst.append(V[0,0])

    # 2. Create random batch of actions.
    u_np_batch = -0.4 + 0.8 * np.random.rand(candidate_num, 4) 
    u_tensor_batch = torch.Tensor(u_np_batch).cuda(cuda_device) # C x 4

    # 3. Compress and predict next reward.
    z_batch = compression(tensor_image_batch) # C x 100
    # z_batch_f = dynamics(torch.cat((z_batch, u_tensor_batch), 1)) # C x 100
    r_batch = reward(torch.cat((z_batch, u_tensor_batch), 1)) # C x 1
    r_batch_np = r_batch.cpu().detach().numpy()

    # 4. Choose index. 
    min_index = torch.argmin(r_batch, 0)
    min_index_np = min_index.cpu().detach().numpy()
    Vhat = V[0,0] + r_batch_np[min_index,:]
    Vhat_lst.append(Vhat[0])

    # 5. apply action.
    best_action = np.squeeze(u_np_batch[min_index_np,:], 0)
    sim.update(best_action)

writer.writerow(V_lst)
writer.writerow(Vhat_lst)

"""
4. Evaluate random CLF controller.
"""
sim.refresh()
traj_length = 32
candidate_num = 3000

V_lst = []
Vhat_lst = []

for i in range(traj_length):
    # 0. Save current image.
    filename = "random_clf/" + "{:03d}".format(i) + ".png"
    cv2.imwrite(os.path.join(image_dir, filename), sim.get_current_image())

    # 1. Feedback on current tensor image.
    tensor_image = image_to_tensor(sim.get_current_image()).cuda() # 1 x 1 x 32 x 32
    tensor_image_batch = tensor_image.repeat(candidate_num, 1, 1, 1) # C x 1 x 32 x 32
    V = lyapunov(tensor_image, device=cuda_device).cpu().detach().numpy()
    V_lst.append(V[0,0])

    # 2. Create random batch of actions.
    u_np_batch = -0.4 + 0.8 * np.random.rand(candidate_num, 4) 
    u_tensor_batch = torch.Tensor(u_np_batch).cuda(cuda_device) # C x 4

    # 3. Compress and predict next reward.
    z_batch = compression(tensor_image_batch) # C x 100
    # z_batch_f = dynamics(torch.cat((z_batch, u_tensor_batch), 1)) # C x 100
    r_batch = reward(torch.cat((z_batch, u_tensor_batch), 1)) # C x 1
    r_batch_np = r_batch.cpu().detach().numpy()

    # 4. Choose random action with a negative reward.
    negative_index_tensor = torch.nonzero(r_batch <= 0, as_tuple=True)
    negative_index_np = negative_index_tensor[0].cpu().detach().numpy()
    action_index = np.random.choice(negative_index_np, 1)
    Vhat = V[0,0] + r_batch_np[action_index,:]
    Vhat_lst.append(Vhat[0,0])

    # 5. apply action.
    best_action = np.squeeze(u_np_batch[action_index,:], 0)
    print(best_action)
    sim.update(best_action)

writer.writerow(V_lst)
writer.writerow(Vhat_lst)

"""
5. Evaluate CEM MPC controller.
"""
sim.refresh()
traj_length = 32 
mpc_window = 3
candidate_num = 10000

V_lst = []
Vhat_lst = []
for i in range(traj_length):
    # 0. Save current image.
    filename = "mpc/" + "{:03d}".format(i) + ".png"
    cv2.imwrite(os.path.join(image_dir, filename), sim.get_current_image())

    # 1. Feedback on current tensor iamge.
    tensor_image = image_to_tensor(sim.get_current_image()).cuda()
    tensor_image_batch = tensor_image.repeat(candidate_num, 1, 1, 1) # C x 1 x 32 x 32
    V = lyapunov(tensor_image, device=cuda_device).cpu().detach().numpy()
    V_lst.append(V[0, 0])

    # 2. Create random batch of actions
    u_np_batch = -0.4 + 0.8 * np.random.rand(candidate_num, 4, mpc_window) # C x 4 x W
    u_tensor_batch = torch.Tensor(u_np_batch).cuda(cuda_device)

    # 3. Run MPC with the predicted model.
    # TODO(terry-suh): this overlaps with definition of num of dimension of z.
    z = compression(tensor_image_batch)
    r_batch = torch.zeros((candidate_num,1)).cuda(cuda_device)
    for k in range(mpc_window):
        u = u_np_batch[:,:,k] # C x 4
        u_tensor_batch = torch.Tensor(u).cuda(cuda_device)
        r_batch += reward(torch.cat((z, u_tensor_batch), 1)) # compute predicted reward
        z = dynamics(torch.cat((z, u_tensor_batch), 1)) # compute next z. 

    r_batch_np = r_batch.cpu().detach().numpy()

    # 4. Choose index. 
    min_index = torch.argmin(r_batch, 0)
    min_index_np = min_index.cpu().detach().numpy()
    Vhat = V[0,0] + r_batch_np[min_index,:]
    Vhat_lst.append(Vhat[0])

    # 5. apply action.
    best_action = np.squeeze(u_np_batch[min_index_np,:,0], 0)
    sim.update(best_action)

writer.writerow(V_lst)
writer.writerow(Vhat_lst)

f.close()
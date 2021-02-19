import numpy as np
import cv2 
import csv 
import os, time 

from sim.onions.onion_sim import OnionSim
from models.onions.rewards import lyapunov, lyapunov_measure 
from models.onions.online_dataloader import ReplayDataset
from models.onions.onion_dataloader import OnionDataset
import models.onions.models as models 
from exp.onions.utils import image_to_tensor 
from PIL import Image 

import torch 
from torch import nn
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from torch.autograd import Variable 
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

"""
We will test the AIS model with feedback in order to do a simple test, which
starts with a simple brute-force greedy controller that looks ahead one step 
into the future and chooses the steepest-descent along Lyapunov. 
"""

"""
0. Setup writer
"""
f = open("/home/hsuh/Documents/p2t_ais/exp/onions/online_rl2.csv", "w")
writer = csv.writer(f, delimiter=',', quotechar='|')
torch_writer = SummaryWriter("runs/online_rl2")

image_dir = "/home/hsuh/Documents/p2t_ais/exp/onions/images/"

"""
1. Setup simulation.
"""
sim = OnionSim()
sim.change_onion_num(100)
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
param_lst = []
model_dir = "/home/hsuh/Documents/p2t_ais/models/onions/weights"
model_name_lst = [
    "reward_mlp_V2.pth",
    "dynamics_mlp_V2.pth",
    "compression_mlp_V2.pth"
]

model_name_lst_new = [
    "reward_mlp_V2.pth",
    "dynamics_mlp_V2.pth",
    "compression_mlp_V2.pth"
]

cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for i in range(len(model_lst)):
    model = model_lst[i]
    model_name = model_name_lst[i]

    model.to(cuda_device)
    param_lst += list(model.parameters())
    try:
        model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
    except FileNotFoundError:
        print("Model " + model_name + "not found.")
        pass
    model.eval()

"""
3. Setup batch dataset and optimizer.
"""
onion_dataset = OnionDataset("data.csv", "images", "/home/hsuh/Documents/p2t_ais/data/onions")
loss = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(param_lst, lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20)

"""
3. Deploy the agent.
"""

trial_length = 500
traj_length = 32
candidate_num = 500

replay_buffer = []
replay_buffer_length = 512
batch_size = 64
learning_rate = 1e-3
num_epochs = 10

for trial in range(trial_length):
    print("Trial: " + str(trial) + " has passed.")
    sim.refresh()

    total_reward = 0.0
    V_lst = []
    Vhat_lst = []

    for i in range(traj_length):

        """
        3.1 Deploy steepest descent policy.
        """
        # 0. Save current image and prepare models to eval mode.
        filename = "steepest_descent/" + "{:03d}".format(i) + ".png"
        cv2.imwrite(os.path.join(image_dir, filename), sim.get_current_image())

        for model in model_lst:
            model.eval()        

        # 1. Feedback on current tensor image.
        image_i = sim.get_current_image()
        tensor_image = image_to_tensor(image_i).cuda() # 1 x 1 x 32 x 32
        tensor_image_batch = tensor_image.repeat(candidate_num, 1, 1, 1) # C x 1 x 32 x 32
        V = lyapunov(tensor_image, device=cuda_device).cpu().detach().numpy()
        V_lst.append(V[0,0])

        # 2. Create random batch of actions.
        u_np_batch = -0.4 + 0.8 * np.random.rand(candidate_num, 4) 
        u_tensor_batch = torch.Tensor(u_np_batch).cuda(cuda_device) # C x 4

        # 3. Compress and predict next reward.
        z_batch = compression(tensor_image_batch) # C x 100
        # z_batch_f = dynamics(torch.cat((z_batch, u_tensor_batch), 1)) # C x 100
        zhat_batch = dynamics(torch.cat((z_batch, u_tensor_batch), 1))
        r_batch = reward(torch.cat((zhat_batch, u_tensor_batch), 1)) # C x 1
        r_batch_np = r_batch.cpu().detach().numpy()

        # Epsilon-greedy.        
        # TODO(terry-suh): get rid of the gross repetition in this code.
        if (np.random.binomial(1, 0.8)):
            # 4. Steepest descent CLF.
            index = torch.argmin(r_batch, 0) # minimize lyapunov?
            index_np = index.cpu().detach().numpy()
            index_np = index_np[0]
        else:
            # 5. Random CLF.
            negative_index_tensor = torch.nonzero((r_batch - torch.Tensor(V).cuda()) <= 0, as_tuple=True)
            negative_index_np = negative_index_tensor[0].cpu().detach().numpy()
            if negative_index_np.size == 0:
                # if there is no action to take, just take a random one.
                index_np = 0
            else:
                # otherwise, choose a random negative one.
                index_np = np.random.choice(negative_index_np, 1)
                index_np = index_np[0]

        Vhat = r_batch_np[index_np,:]
        Vhat_lst.append(Vhat[0])


        # 5. apply action.
        best_action = u_np_batch[index_np,:]
        sim.update(best_action)

        # 6. Observe next image and compute actual reward. 
        image_f = sim.get_current_image()
        tensor_image_f = image_to_tensor(image_f).cuda()
        Vnext = lyapunov(tensor_image_f, device=cuda_device).cpu().detach().numpy()

        """
        3.2 Add SARS samples to Replay Buffer.
        """
        Si = image_i
        A = best_action 
        R = V
        Sf = image_f

        replay_buffer.append([Si, A, R, Sf])
        if len(replay_buffer) > replay_buffer_length:
            replay_buffer.pop(0)

        total_reward += R

    torch_writer.add_scalar("performance", total_reward, trial)

    """
    3.3 Run online SGD to refine the model. 
    """
    # If the replay buffer is full, draw at random to do SGD on batch or replay.
    if (len(replay_buffer) == replay_buffer_length):
        print("Replay buffer full. Entering online training mode.")
        # 70 percent chance of doing SGD on replay buffer instead of batch.
        if (np.random.binomial(1, 0.9)):
            dataloader = DataLoader(ReplayDataset(replay_buffer), batch_size=batch_size, shuffle=True, num_workers=24)
            num_epochs = 30
        else:
            dataloader = DataLoader(onion_dataset, batch_size=batch_size, shuffle=True, num_workers=24)
            num_epochs = 10

        for epoch in range(num_epochs):
            for model in model_lst:
                model.train()
                for g in optimizer.param_groups:
                    g['lr'] = 1e-3

            running_loss = 0.0

            for data in tqdm(dataloader):
                image_i = Variable(data['image_i']).cuda(cuda_device)
                image_f = Variable(data['image_f']).cuda(cuda_device)
                u = Variable(data['u']).cuda(cuda_device).float()

                optimizer.zero_grad()

                z_i = compression(image_i)
                z_f = compression(image_f)

                rhat = reward(torch.cat((z_i, u), 1))
                zhat_f = dynamics(torch.cat((z_i, u), 1))
                z_loss = loss(zhat_f, z_f)

                rtrue = lyapunov(image_i, device=cuda_device)
                r_loss = loss(rhat, rtrue)

                weight = 0.05
                total_loss = weight * r_loss + (1 - weight) * z_loss
                running_loss += total_loss 

                total_loss.backward()
                optimizer.step()

            print('epoch[{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, total_loss.item()))
            torch_writer.add_scalar('training_loss', -total_loss.item(), epoch)

            if (epoch >= 15):
                for g in optimizer.param_groups:
                    g['lr'] = 1e-4

        if (trial % 100 == 99):
            # TODO(terry-suh): save weights
            for i in range(len(model_lst)):
                model = model_lst[i]
                model_name = model_name_lst_new[i]
                torch.save(model.state_dict(), os.path.join(model_dir, model_name))
            print("Model Saved!")

    else:
    # If not, do nothing. keep collecting data.
    # TODO(terry-suh): it's possible to do additional SGDs....
        pass

    writer.writerow(V_lst)
    writer.writerow(Vhat_lst)

f.close()
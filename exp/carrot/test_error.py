import numpy as np
import csv, os 
import matplotlib.pyplot as plt

import torch 
from torch import nn
from sim.onions.onion_sim import OnionSim
from models.onions.rewards import lyapunov
from exp.onions.utils import plot_arrow, image_to_tensor 
import models.onions.models as models

def setup_models(comp, dyn, model_name_lst):
    z = 100 
    a = 4 

    reward = models.RewardMLP(z, a)
    if (comp == "mlp"):
        compression = models.CompressionMLP(z, a)
    elif (comp == "conv"):
        compression = models.CompressionConv(z, a)
    elif (comp == "linear"):
        compression = models.CompressionLinear(z, a)
    else:
        raise ValueError("compression doesn't support this model.")

    if (dyn == "mlp"):
        dynamics = models.DynamicsMLP(z, a)
    elif (dyn == "linear"):
        dynamics = models.DynamicsLinear(z, a)
    else:
        raise ValueError("dynamics doesn't support this model.")

    model_lst = [reward, dynamics, compression]
    model_dir = "/home/hsuh/Documents/p2t_ais/models/onions/weights"

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

    return model_lst

def print_statistics(model_name, error_lst):
    print("====Statistics for " + model_name + " ====")
    print("Error mean: " + str(np.mean(error_lst)))
    print("Error  std: " + str(np.std(error_lst)))
    plt.figure()
    plt.hist(error_lst, bins=range(0, 500, 20))
    plt.show()

def get_test_error(model_name, comp, dyn, model_name_lst):
    reward, dynamics, compression = setup_models(comp, dyn, model_name_lst)

    sim = OnionSim()
    sim.refresh()

    # Sample a random input trajectory open-loop.
    trial_length = 10
    traj_length = 10

    error_lst = []

    for i in range(trial_length):
        for j in range(traj_length):
            u = -0.4 + 0.8 * np.random.rand(4)
            u_tensor = torch.unsqueeze(torch.Tensor(u).cuda(), 0)

            tensor_image = image_to_tensor(sim.get_current_image()).cuda()
            # TODO(terry-suh): clean up here so the device isn't hard-coded.
            V_now = lyapunov(tensor_image, device="cuda:0")

            z_i = compression(tensor_image)
            z_hat = dynamics(torch.cat((z_i, u_tensor), 1))
            rhat = reward(torch.cat((z_i, u_tensor), 1))

            sim.update(u)
            next_image = image_to_tensor(sim.get_current_image()).cuda()
            V_next = lyapunov(next_image, device="cuda:0")

            z_f = compression(next_image)

            weight = 0.9
            loss = nn.MSELoss(reduction='sum')

            #print(V_now.shape)
            r_loss = loss(rhat, V_next - V_now)
            z_loss = loss(z_hat, z_f)
            #print(r_loss, z_loss)
            total_loss = weight * r_loss + (1 - weight) * z_loss

            total_loss_np = total_loss.cpu().detach().numpy()
            error_lst.append(total_loss_np.item())

        sim.change_onion_num(sim.onion_num - 20)

    print_statistics(model_name, error_lst)

    return error_lst

error_lst = get_test_error("full_linear", "linear", "linear", [
    "reward_mlp_lincomp_lindyn.pth",
    "dynamics_mlp_lincomp_lindyn.pth",
    "compression_lincomp_lindyn.pth"])




    


    
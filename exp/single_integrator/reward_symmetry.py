import gym 
import os
import yaml, argparse, importlib
from models import model_io
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2


"""
0. Argparse directory yaml file. Defaults to config.yaml.
"""
parser = argparse.ArgumentParser(description="Load config file: ")
parser.add_argument("-c", "--config", help=
    "configuration file. example: config/default.yaml", required=True)
args = parser.parse_args()

torch.cuda.empty_cache()
cuda_device = ("cuda:0" if torch.cuda.is_available() else "cpu")

"""
1. Load and setup models.
"""
config = yaml.load(open(args.config, 'r'))
num_z = config["model"]["z"]
num_a = config["model"]["a"]

model_module = importlib.import_module("models." + config["env"] + ".model")

reward = getattr(model_module,
    config["model"]["reward"])(num_z, num_a)
dynamics = getattr(model_module,
    config["model"]["dynamics"])(num_z, num_a)
compression = getattr(model_module,
    config["model"]["compression"])(num_z, num_a)

model_dict = {
    "reward": reward,
    "dynamics": dynamics,
    "compression": compression
}

model_dir = os.path.join(config["path"], config["load_model"]["model_dir"])
model_load_dict = config["load_model"]
param_lst = model_io.load_models(
    model_dict, model_load_dict, model_dir, cuda_device=cuda_device)

"""
1. Load gym environment and models.
"""
env = gym.make("SingleIntegrator-v0")

current_image = env.reset() 
traj_lst = []

mode = 0
for i in range(500):
    current_pos = np.array(list(env.sim.body.position))
    rel_pos = np.array(list(env.sim.body.position)) - np.array([250, 250])

    # Ciruclar 
    #velocity = np.array([rel_pos[1], -rel_pos[0]]) / 1000.0
    # Straight

    if (mode == 0):
        velocity = -0.3 * np.array([rel_pos[0], rel_pos[1]]) / np.linalg.norm(rel_pos)
        if np.linalg.norm(rel_pos) < 20.0:
            mode = 1 
    if (mode == 1):
        velocity = 0.3 * np.array([rel_pos[0], rel_pos[1]]) / np.linalg.norm(rel_pos)
        if np.linalg.norm(rel_pos) > 200.0:        
            mode = 0

    cv2.imwrite(
        os.path.join(config["path"],
        "exp/single_integrator/images/reward_symmetry_line/image/" + "{:04d}".format(i) + ".png"), env.render())

    current_z = compression(torch.Tensor(current_image).cuda().unsqueeze(0))
    current_z = current_z.detach().cpu().numpy()[0]
    print(current_z)
    traj_lst.append(current_z)

    next_image, reward, done, info = env.step(velocity)

    current_image = next_image

    plt.figure()
    traj_lst_np = np.array(traj_lst)
    plt.plot(traj_lst_np[:,0], traj_lst_np[:,1], "k-")
    plt.plot(current_z[0], current_z[1], "ro")
    plt.xlim([-1.0, 1.0])
    plt.ylim([-1.0, 1.0])
    plt.savefig(
        os.path.join(config["path"],
        "exp/single_integrator/images/reward_symmetry_line/plot/" + "{:04d}".format(i) + ".png"))
    plt.close()

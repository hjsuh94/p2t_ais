import numpy as np
import cv2
import csv
import os, time
import yaml, argparse
import gym

import models
from models import model_io
import importlib
from policies.policy import Sampling_MPC


"""
Script to collect data for pile simulation.
"""

def write_image(filename, image):
    # Black and white image, just save it.
    if (len(image.shape) == 2):
        cv2.imwrite(filename, image)
    # Color image, need to switch color schemes.
    else:
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        

"""
0. Argparse directory yaml file. Defaults to config.yaml.
"""
parser = argparse.ArgumentParser(description="Load config file: ")
parser.add_argument("-c", "--config", help=
    "configuration file. example: config/default.yaml", required=True)
args = parser.parse_args()

"""
1. Load settings for the file and setup writers.
"""
config = yaml.load(open(args.config, "r"))
env = gym.make(config["env_name"])

root_dir = os.path.join(config["path"], config["dataset"]["data_dir"])
image_dir = os.path.join(root_dir, config["dataset"]["image_dir"])
data_dir = os.path.join(root_dir, config["dataset"]["file"])

f = open(data_dir, 'w')
writer = csv.writer(f, delimiter=',', quotechar='|')

"""
2. Setup models if necessary.
"""
if (config["dataset"]["policy"] is not "random"):
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
        model_dict, model_load_dict, model_dir, cuda_device="cuda") 

"""
2. Start Collecting Data.
"""
for episode in range(config["dataset"]["num_episodes"]):
    image = env.reset()
    episode_dir = os.path.join(image_dir, "{:04d}".format(episode))
    os.mkdir(episode_dir)

    write_image(os.path.join(episode_dir, "0000.png"), image)

    for count in range(config["dataset"]["episode_length"]):
    
        if (config["dataset"]["policy"] == "random"):
            action = env.action_space.sample()
        # TODO(terry-suh): implement fixed action cases.
        elif (config["dataset"]["policy"] == "sampling_mpc"):
            result = Sampling_MPC(image /255., model_dict, 3, 5000,
                env.action_space.low, env.action_space.high, discount=0.8)
            action = result["action"]
        elif (config["dataset"]["policy"] == "sampling_clf"):
            action = Sampling_CLF(image/255., model_dict, 5000, 
                env.action_space.low, env.action_space.high)

        image, reward, done, info = env.step(action)

        if (done):
            env.reset()
            break

        filename_prev = "{:04d}".format(count) + ".png"
        filename_next = "{:04d}".format(count + 1) + ".png"

        # save current screenshot. 
        # Black and white cases.
        write_image(os.path.join(episode_dir, filename_next), image)

        # file data into masterfile.
        logrow = [episode, filename_prev, filename_next, reward]
        logrow += list(map(str, action))
        writer.writerow(logrow)
    
f.close()

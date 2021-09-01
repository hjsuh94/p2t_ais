import numpy as np
import csv 
import yaml, argparse 
import pandas as pd
import os, time
import json

"""
Script to change equation error sim to simulation error sim
"""

parser = argparse.ArgumentParser(description="data file: ")
parser.add_argument("-c", "--config", help=
    "configuration file. example: config/default.yaml", required=True)
args = parser.parse_args()

config = yaml.load(open(args.config, "r"))

num_episodes = config["dataset"]["num_episodes"]
episode_length = config["dataset"]["episode_length"]
horizon = config["dataset"]["horizon"]

root_dir = os.path.join(config["path"], config["dataset"]["data_dir"])
image_dir = os.path.join(root_dir, config["dataset"]["image_dir"])
data_dir = os.path.join(root_dir, config["dataset"]["file"])

data_file = pd.read_csv(data_dir, header=None)

stop_loop = False
storage_lst = []
count = 0
while(True):
    within_episode = True
    for t in range(horizon):
        try:
            within_episode = within_episode and (data_file.iloc[count,0] == data_file.iloc[count + t, 0])
        except IndexError:
            stop_loop = True
            break
        
    if (stop_loop):
        break

    if within_episode:
        episode_num = data_file.iloc[count,0]
        current_image = data_file.iloc[count,1]
        reward_traj = np.array(data_file.iloc[count:count+horizon, 3])
        input_traj = np.array(data_file.iloc[count:count+horizon, 4:8])

        storage_lst.append([episode_num, current_image, reward_traj, input_traj])

    else:
        print(episode_num)

    count += 1 

save_filename = os.path.join(root_dir, "simulation_lst_4.npy")
np.save(save_filename, storage_lst, allow_pickle=True)
#with open(os.path.join(root_dir, "simulation_lst.json"), "w") as f:
#    json.dump(storage_lst, f)




import numpy as np
import cv2
import csv
import os, time
import yaml, argparse
import gym

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
2. Start Collecting Data.
"""
for episode in range(config["dataset"]["num_episodes"]):
    current_image = env.reset()
    episode_dir = os.path.join(image_dir, "{:04d}".format(episode))
    os.mkdir(episode_dir)

    write_image(os.path.join(episode_dir, "0000.png"), current_image)

    for count in range(config["dataset"]["episode_length"]):
    
        if (config["dataset"]["policy"] == "random"):
            action = env.action_space.sample()
        # TODO(terry-suh): implement fixed action cases.
        elif (config["dataset"]["policy"] == "fixed"):
            action = np.array([-0.4, -0.4, 0.2, 0.2])

        next_image, reward, done, info = env.step(action)

        filename_prev = "{:04d}".format(count) + ".png"
        filename_next = "{:04d}".format(count + 1) + ".png"

        # save current screenshot. 
        # Black and white cases.
        write_image(os.path.join(episode_dir, filename_next), next_image)

        # file data into masterfile.
        logrow = [episode, filename_prev, filename_next, reward]
        logrow += list(map(str, action))
        writer.writerow(logrow)
    
f.close()
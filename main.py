import yaml, argparse 
import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
from models import model_io, train
import importlib

"""
0. Argparse directory yaml file. Defaults to config.yaml.
"""
parser = argparse.ArgumentParser(description="Load config file: ")
parser.add_argument("-c", "--config", help=
    "configuration file. example: config/default.yaml", required=True)
args = parser.parse_args()

"""
1. Load settings for the file and setup writers. config.yaml acts as argparse.
"""
config = yaml.load(open(args.config, "r"))
torch_writer = SummaryWriter(os.path.join(config["path"], "runs/" + 
    config["tensorboard_dir"]))
torch.cuda.empty_cache()
cuda_device = ("cuda:0" if torch.cuda.is_available() else "cpu")

"""
2. Load and setup models.
"""
num_z = config["model"]["z"]
num_a = config["model"]["a"]

# NOTE(terry-suh): This loads class names from config files and initatializes 
# these classes.

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
# TODO(terry-suh): get rid of repetition.
model_load_dict = config["load_model"]

# NOTE(terry-suh): this relies on specific order of reward-dynamics-comp across
# model_lst and model_load_lst.
# TODO(terry-suh): change this to a dict implementation.
param_lst = model_io.load_models(
    model_dict, model_load_dict, model_dir, cuda_device=cuda_device)

"""
3. Hyperparameters and training settings.
"""
dataset_module = importlib.import_module("models." + config["env"] + ".offline_dataloader")
if (config["mode"] != "deploy"):
    offline_dataset = getattr(dataset_module, "OfflineDataset")(
        config["dataset"]["file"],
        config["dataset"]["image_dir"],
        os.path.join(config["path"], config["dataset"]["data_dir"])
    )
    offline_dataloader = DataLoader(offline_dataset,
        batch_size=config["offline_train"]["batch_size"],
        shuffle=True, num_workers=24)

    print(config["offline_train"]["initial_lr"])

    optimizer = torch.optim.Adam(param_lst, 
        lr=config["offline_train"]["initial_lr"], weight_decay=1e-5)
    lr_scheduler = getattr(torch.optim.lr_scheduler, config["offline_train"]["lr_scheduler"])
    scheduler  = lr_scheduler(optimizer, step_size=config["offline_train"]["lr_step_size"])

"""
4. Execute main file.
"""
if (config["mode"] == "train"):
    train.train(
        model_dict, config["offline_train"]["lmbda"],
        offline_dataloader, optimizer, config["offline_train"]["num_epochs"],
        scheduler, torch_writer, config["save_model"],
        os.path.join(config["path"], config["save_model"]["model_dir"]), save='best')

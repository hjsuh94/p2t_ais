import yaml, argparse 
import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
from models import model_io, train, evaluate, interpret, online_rl, evaluate_ensemble, online_rl_ensemble
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
cuda_device = config["device"]

"""
2. Load and setup models.
"""
num_z = config["model"]["z"]
num_a = config["model"]["a"]
num_models = config["model"]["num_models"]

# NOTE(terry-suh): This loads class names from config files and initatializes 
# these classes.

model_module = importlib.import_module("models." + config["env"] + ".model")

if (num_models == 1):
    reward = getattr(model_module,
        config["model"]["reward"])(num_z, num_a)
    dynamics = getattr(model_module,
        config["model"]["dynamics"])(num_z, num_a)
    compression = getattr(model_module,
        config["model"]["compression"])(num_z, num_a)
else:
    reward = getattr(model_module,
        config["model"]["reward"])(num_z, num_a, num_models)
    dynamics = getattr(model_module,
        config["model"]["dynamics"])(num_z, num_a, num_models)
    compression = getattr(model_module,
        config["model"]["compression"])(num_z, num_a, num_models)

model_dict = {
    "reward": reward,
    "dynamics": dynamics,
    "compression": compression
}

if "decompression" in config["model"]:
    decompression = getattr(model_module,
        config["model"]["decompression"])(num_z, num_a)
    model_dict["decompression"] = decompression

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
if (config["mode"] == "train"):
    if (config["offline_train"]["mode"] == "equation"):
        offline_dataset1 = getattr(dataset_module, "OfflineDataset")(
            config["dataset"]["file"],
            config["dataset"]["image_dir"],
            os.path.join(config["path"], config["dataset"]["data_dir"])
        )

        """
        offline_dataset2 = getattr(dataset_module, "OfflineDataset")(
            "data.csv",
            "images",
            os.path.join(config["path"], "data/carrot_online")
        )
        """

        #offline_dataset = torch.utils.data.ConcatDataset([offline_dataset1, offline_dataset2])

    elif (config["offline_train"]["mode"] == "simulation"):
        offline_dataset = getattr(dataset_module, "SimulationDataset")(
            "simulation_lst_4.npy",
            config["dataset"]["image_dir"],
            os.path.join(config["path"], config["dataset"]["data_dir"])
        )
    else:
        raise ValueError("Wrong mode for trainig. Only equation / simulation is supported.")
       
    offline_dataloader = DataLoader(offline_dataset1,
        batch_size=config["offline_train"]["batch_size"],
        shuffle=True, num_workers=24)

    optimizer = torch.optim.Adam(param_lst, 
        lr=config["offline_train"]["initial_lr"])
    lr_scheduler = getattr(torch.optim.lr_scheduler, config["offline_train"]["lr_scheduler"])
    if (config["offline_train"]["lr_scheduler"] == "StepLR"):
        scheduler = lr_scheduler(optimizer, step_size=config["offline_train"]["lr_step_size"])
    else:
        scheduler = lr_scheduler(optimizer, patience=20)

"""
4. Execute main file.
"""
if (config["mode"] == "train"):
    train.train(
        model_dict, config["offline_train"], config,
        offline_dataloader, optimizer, scheduler, torch_writer, save="best", cuda_device="cuda:1")

elif (config["mode"] == "evaluate"):
    if (num_models == 1):
        evaluate.evaluate(model_dict, config)
    else:
        evaluate_ensemble.evaluate_ensemble(model_dict, config)

elif (config["mode"] == "online_rl"):  
    if (num_models == 1):
        online_rl.online_rl(model_dict, config, param_lst, torch_writer)
    else:
        
        online_rl_ensemble.online_rl_ensemble(model_dict, config, param_lst, torch_writer)


else:
    raise ValueError("Wrong mode.")
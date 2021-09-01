import torch.nn as nn 
import torch
import torch.nn.functional as F 
import numpy as np 
import os

"""
5. Load and Save Models.
"""
def load_models(model_dict, model_name_dict, model_dir, cuda_device="cuda:0"):
    """
    5.1 Loads models from a given list of models and their names.
    """
    param_lst = []

    key_lst = ["reward", "dynamics", "compression"]

    if "decompression" in model_dict:
        key_lst.append("decompression")
    
    for key in key_lst:
        model = model_dict[key]
        model_name = model_name_dict[key]
        model.to(cuda_device)
        param_lst += list(model.parameters())

        try:
            model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
        except FileNotFoundError:
            print("Model " + model_name + " not found.")
        model.eval()

    return param_lst

def save_models(model_dict, model_name_dict, model_dir):
    """
    5.2 Save models from a given list of models and their names.
    """ 
    key_lst = ["reward", "dynamics", "compression"]

    if "decompression" in model_dict:
        key_lst.append("decompression")

    for key in key_lst:
        model = model_dict[key]
        model_name = model_name_dict[key]
        torch.save(model.state_dict(), os.path.join(model_dir, model_name))
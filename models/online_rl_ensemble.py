import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import gym
import importlib, os

import torch 
from torch.utils.data import DataLoader, ConcatDataset

import models.train
from policies.policy import rollout_ensemble, Sampling_MPC_Variance, Sampling_CLF

def online_rl_ensemble(model_dict, config, param_lst, torch_writer):

    # 1. Setup gym environment.
    env = gym.make(config["env_name"])

    # 2. Setup config file. 
    num_episodes = config["online_train"]["num_episodes"]
    buffer_size = config["online_train"]["buffer_size"]
    training_frequency = config["online_train"]["training_frequency"]
    episode_length = config["online_train"]["episode_length"]

    policy = config["online_train"]["policy"]
    horizon = config["online_train"]["policy_horizon"]
    num_candidates = config["online_train"]["policy_num_candidates"]
    discount = config["online_train"]["policy_discount"]

    """
    3. Setup offline dataset and optimizer.
    """
    offline_dataset_module = importlib.import_module("models." + config["env"] + ".offline_dataloader")

    # only support equation error for now.
    offline_dataset = getattr(offline_dataset_module, "OfflineDataset")(
        config["dataset"]["file"],
        config["dataset"]["image_dir"],
        os.path.join(config["path"], config["dataset"]["data_dir"])
    )

    """
    # TODO(terry-suh): move this to a config file / main.py. Too lazy....
    offline_dataset2 = getattr(offline_dataset_module, "OfflineDataset")(
        "data.csv",
        "images",
        os.path.join(config["path"], "data/carrot_online")
    )

    offline_dataset = torch.utils.data.ConcatDataset([offline_dataset1, offline_dataset2])
    """

    offline_dataloader = DataLoader(offline_dataset, 
        batch_size=config["offline_train"]["batch_size"],
        shuffle=True, num_workers=24)

    offline_optimizer = torch.optim.Adam(param_lst, lr=config["offline_train"]["initial_lr"])
    offline_lr_scheduler = getattr(torch.optim.lr_scheduler, config["offline_train"]["lr_scheduler"])
    if (config["offline_train"]["lr_scheduler"] == "StepLR"):
        offline_scheduler = offline_lr_scheduler(offline_optimizer, step_size=config["offline_train"]["lr_step_size"])
    else:
        offline_scheduler = offline_lr_scheduler(optimizer, patience=20)

    """
    4. Setup online dataset and optimizer.
    """
    online_dataset_module = importlib.import_module("models." + config["env"] + ".online_dataloader")

    """
    5. Start filling the replay buffer.
    """    
    replay_buffer = []
    training_count = 0
    
    for episode in range(num_episodes):
        image_log = []
        reward_log = []
        action_log = []
        mpc_log = []

        image = env.reset() / 255. 
        image_log.append(image)

        reward_sum = 0.0

        for t in range(episode_length):

            if (policy == "random"):
                action = env.action_space.sample()
            elif (policy == "sampling_mpc"):
                result = Sampling_MPC_Variance(image, model_dict, horizon, num_candidates,
                    env.action_space.low, env.action_space.high, discount=discount, device=config["device"])
                action = result["action"]
                mpc_log.append(result["r_mat"])
            elif (policy == "sampllng_clf"):
                result = Sampling_CLF(image, model_dict, num_candidates,
                    env.action_space.low, env_action_space.high)

            SARS = []
            SARS.append(image)
            SARS.append(action)

            image, reward, done, info = env.step(action)
            image = image / 255.

            SARS.append(reward)
            SARS.append(image)

            replay_buffer.append(SARS)

            reward_log.append(reward)
            action_log.append(action)
            image_log.append(image)

            reward_sum += np.power(discount, t) * reward

            """
            6. Once the buffer is full, train based on frequency.
            """

            if (len(replay_buffer) > buffer_size):
                replay_buffer.pop(0)
                training_count += 1

                if (training_count > training_frequency):
                    training_count = 0

                    online_dataset = getattr(online_dataset_module, "OnlineDataset")(replay_buffer)

                    combined_dataset = torch.utils.data.ConcatDataset([online_dataset, offline_dataset])

                    dataloader = DataLoader(combined_dataset,
                        batch_size=config["online_train"]["batch_size"],
                        shuffle=True, num_workers=24)

                    online_optimizer = torch.optim.Adam(param_lst, lr=config["online_train"]["initial_lr"])
                    online_lr_scheduler = getattr(torch.optim.lr_scheduler, config["online_train"]["lr_scheduler"])
                    if (config["online_train"]["lr_scheduler"] == "StepLR"):
                        online_scheduler = online_lr_scheduler(online_optimizer, step_size=config["online_train"]["lr_step_size"])
                    else:
                        online_scheduler = online_lr_scheduler(online_optimizer, patience=20)

                    models.train.train(
                        model_dict, config["online_train"], config, 
                        dataloader, online_optimizer, online_scheduler, torch_writer, save="best",
                        write_results=False, cuda_device=config["device"])


        """
        7. Backtest every episode.
        """

        image_log = np.array(image_log)
        action_log = np.array(action_log)
        reward_log = np.array(reward_log)
        mpc_log = np.array(mpc_log)

        rhat_log = np.zeros((config["model"]["num_models"], episode_length - horizon, horizon))

        episode_simulation_error = []

        for t in range(episode_length - horizon):
            image = np.expand_dims(image_log[t], axis=0)
            image = np.expand_dims(image, axis=0)

            action_seq = action_log[t:t + horizon]
            action_seq = np.expand_dims(action_seq, axis=0)

            result = rollout_ensemble(image, model_dict, horizon, action_seq, discount=discount)
            rhat_log[:,t,:] = np.squeeze(result["r_mat"].detach().cpu().numpy(), 1)

            discounted_simulation_error = 0.0
            for h in range(horizon):
                mean_rhat = np.mean(rhat_log[:,t,h])
                discounted_simulation_error += np.power(discount, h) * np.abs(reward_log[t+h] - mean_rhat)

            episode_simulation_error.append(discounted_simulation_error)

        episode_simulation_error = np.array(episode_simulation_error)

        log_dir = os.path.join(config["path"], config["online_train"]["log_dir"])
        pred_plot_dir = os.path.join(log_dir, "plots_pred")
        mpc_plot_dir = os.path.join(log_dir, "plots_mpc")


        plt.figure()
        cmap = cm.get_cmap('magma')
        for t in range(episode_length - horizon):
            normalized_t = float(t) / (episode_length - horizon)

            mean = np.mean(rhat_log[:,t,:], axis=0)
            var=  np.std(rhat_log[:,t,:], axis=0)

            plt.fill_between(range(t, t+horizon), mean-var, mean+var, color=cmap(normalized_t), alpha=0.5)

        plt.plot(range(episode_length), np.array(reward_log), 'o-', color='lime', alpha=0.5)

        plt.xlabel('Time')
        plt.ylabel('Reward')
        plt.title("Backtest Simulation Error")
        plt.savefig(os.path.join(pred_plot_dir, "{:05d}".format(episode)))

        plt.figure()
        cmap = cm.get_cmap('twilight')
        for t in range(episode_length - horizon):
            normalized_t = float(t) / (episode_length - horizon)

            mean = np.mean(mpc_log[t,:,:], axis=0)
            var=  np.std(mpc_log[t,:,:], axis=0)

            plt.fill_between(range(t, t+horizon), mean-var, mean+var, color=cmap(normalized_t), alpha=0.5)

        plt.plot(range(episode_length), np.array(reward_log), 'o-', color='lime', alpha=0.5)

        plt.xlabel('Time')
        plt.ylabel('Reward')
        plt.title("MPC Plans")
        plt.savefig(os.path.join(mpc_plot_dir, "{:05d}".format(episode)))        
        
        torch_writer.add_scalar("performance", reward_sum, episode)
        torch_writer.add_scalar("prediction", np.mean(episode_simulation_error), episode)

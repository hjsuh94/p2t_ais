import numpy as np 
import os
import gym
from policies.policy import rollout, Sampling_MPC, Sampling_CLF, rollout_batch
import matplotlib.pyplot as plt 
from matplotlib import cm
import cv2

import torch

def evaluate(model_dict, config):
    """
    Evaluate the performance of model across various settings.
    Metric is, in general, simulation error of predicting rewards.
    """

    # 1. Setup gym environment.
    env = gym.make(config["env_name"])

    # 2. Setup folder to save graphs and images.
    log_dir = os.path.join(config["path"], config["evaluate"]["log_dir"])
    pred_plot_dir = os.path.join(log_dir, "plots_pred")
    mpc_plot_dir = os.path.join(log_dir, "plots_mpc")    
    image_dir = os.path.join(log_dir, "images")

    # 3. Repeat for episode and rollout.
    num_episodes = config["evaluate"]["num_episodes"]
    episode_length = config["evaluate"]["episode_length"]
    horizon = config["evaluate"]["horizon"]
    discount = config["evaluate"]["discount"]
    policy = config["evaluate"]["policy"]
    num_candidates = config["evaluate"]["num_candidates"]

    total_log = np.zeros(((episode_length - horizon) * num_episodes, horizon))
    perf_stat = []

    for episode in range(num_episodes):

        image_log = np.zeros(tuple([episode_length + 1] + list(env.observation_space.shape)))
        action_log = np.zeros(tuple([episode_length] + list(env.action_space.shape)))
        reward_log = np.zeros(episode_length)
        mpc_log = []

        episode_dir = os.path.join(image_dir, "{:02d}".format(episode))
        os.mkdir(episode_dir)

        # 4. Before using our model, we will have a full record of a run so we can backtest.
        image = env.reset() / 255.
        image_log[0] = image

        reward_sum = 0.0

        for t in range(episode_length):
            filename = "{:02d}".format(t) + ".png"
            image_name = os.path.join(episode_dir, filename)
            cv2.imwrite(image_name, env.render())


            if (policy == "random"):
                action = env.action_space.sample()
            elif (policy == "sampling_mpc"):
                result = Sampling_MPC(image, model_dict, horizon, num_candidates,
                    env.action_space.low, env.action_space.high)

                action = result["action"]
                mpc_log.append(result["r_mat"])

            elif (policy == "sampling_clf"):
                action = Sampling_CLF(image, model_dict, num_candidates,
                    env.action_space.low, env.action_space.high)

            image, reward, done, info = env.step(action)
            image = image / 255.

            image_log[t + 1] = image
            reward_log[t] = reward
            action_log[t] = action

            reward_sum += np.power(discount, t) * reward

        perf_stat.append(reward_sum)
        mpc_log = np.array(mpc_log)

        # 5. Now we have a full record of the run, so we can roll out and backtest.

        rhat_log = np.zeros((episode_length - horizon, horizon))
        for t in range(episode_length - horizon):

            image = np.expand_dims(image_log[t], axis=0)
            image = np.expand_dims(image, axis=0)            
            action_seq = action_log[t:t + horizon]

            result = rollout(image, model_dict, horizon, action_seq, discount=discount)

            rhat_log[t,:] = result["r_mat"]

            # Log the absolute difference here.
            discounted_simulation_error = 0.0
            for h in range(horizon):
                discounted_simulation_error += np.power(discount, h) * np.abs(reward_log[t+h] - rhat_log[t,h])

            total_log[(episode_length - horizon) * episode + t, :] =  discounted_simulation_error

        # 6. Now that we have everything, plot our simulation error.
        plt.figure()

        cmap = cm.get_cmap('magma')

        for t in range(episode_length - horizon):
            normalized_t = float(t) / (episode_length - horizon)
            #plt.plot(range(t, t  + horizon), rhat_log[t,:], '-',)
            plt.plot(range(t, t  + horizon), rhat_log[t,:], 'o-', color=cmap(normalized_t), alpha=1.0)

        plt.plot(range(episode_length), reward_log, 'o-', color='lime', alpha=0.5)

        plt.xlabel('Time')
        plt.ylabel('Reward')
        plt.title('Model Prediction Error')
        plt.savefig(os.path.join(pred_plot_dir, "{:02d}".format(episode)))

        # 7. If using mpc, also plot the mpc-imagined trajectories.
        if (policy == "sampling_mpc"):
            plt.figure()
            cmap = cm.get_cmap('twilight')

            for t in range(episode_length - horizon):
                normalized_t = float(t) / (episode_length - horizon)            
                plt.plot(range(t, t + horizon), mpc_log[t,:], 'o-', color=cmap(normalized_t), alpha=1.0)

            plt.plot(range(episode_length), reward_log, 'o-', color='lime', alpha=0.5)
            plt.xlabel('Time')
            plt.ylabel('Reward')
            plt.title('MPC Plans')
            plt.savefig(os.path.join(mpc_plot_dir, "{:02d}".format(episode)))        


    # 7. Now produce some statistics about the performance of the overall test.
    total_stat = np.zeros((episode_length - horizon) * num_episodes)
    for h in range(horizon):
        total_stat += np.power(discount, h) * total_log[:,h]

    pred_mean = np.mean(total_stat)
    pred_std = np.std(total_stat)

    perf_stat = np.array(perf_stat)
    perf_mean = np.mean(perf_stat)
    perf_std = np.std(perf_stat)

    filename = os.path.join(log_dir, "eval.txt")
    f = open(filename, 'w')
    f.write("Evaluation Results:\n")
    f.write("Prediction Mean:  " + "{:06f}".format(pred_mean) + "\n")
    f.write("Prediction STD:   " + "{:06f}".format(pred_std) + "\n")
    f.write("Performance Mean: " + "{:06f}".format(perf_mean) + "\n")
    f.write("Performance STD : " + "{:06f}".format(perf_std) + "\n")    

    f.close()

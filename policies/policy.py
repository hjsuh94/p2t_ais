import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

def rollout(obs, models, horizon, action_seq, discount=1.0, device="cuda:0"):
    """
    Simple rollout function of the model based on 
    @ params obs (np.array): observation at current time. 
    @ params model (dict): dictionary of torch NNs with keys {"compression", "reward", "dynamics"}.
    @ params horizon (int): number of horizons to rollout.
    @ params action_seq (np.array): T x dim(a) array where action_seq[t,:] is the action.
    """
    # 1. Parse dictionary and setup models.
    compression = models["compression"].cuda(device)
    reward = models["reward"].cuda(device)
    dynamics = models["dynamics"].cuda(device)

    # 2. Compress the first observation
    obs_tensor = torch.Tensor(obs).cuda(device)
    z_i = compression(obs_tensor)

    dim_z = z_i.shape[1]
    action_seq_tensor = torch.Tensor(action_seq).cuda(device)

    # 3. Set up storages
    r_mat = torch.Tensor(horizon).cuda(device)
    z_mat = torch.Tensor(horizon + 1, dim_z).cuda(device)
    z_mat[0,:] = z_i

    cumulative_reward = 0.0

    # 4. Rollout model 
    for t in range(horizon):
        u_t = action_seq_tensor[t,:].unsqueeze(0)
        z_t = z_mat[t,:].unsqueeze(0)

        zu_t = torch.cat((z_t, u_t), 1)

        r_mat[t] = reward(zu_t).squeeze(1)
        z_mat[t+1,:] = dynamics(zu_t)#.squeeze(0)
        cumulative_reward += np.power(discount, t) * r_mat[t]

    # 5. Return results
    result = {
        "r_mat": r_mat.detach().cpu().numpy(),
        "z_mat": z_mat.detach().cpu().numpy(),
        "r_sum": cumulative_reward.detach().cpu().numpy(),
    }    

    return result 



def rollout_batch(obs, models, horizon, action_seq, discount=1.0, device="cuda:0"):
    """
    Simple rollout function of the model based on 
    @ params obs (np.array): observation at current time. 
    @ params model (dict): dictionary of torch NNs with keys {"compression", "reward", "dynamics"}.
    @ params horizon (int): number of horizons to rollout.
    @ params action_seq (np.array): T x dim(a) array where action_seq[t,:] is the action.
    """
    # 1. Parse dictionary and setup models.
    compression = models["compression"].cuda(device)
    reward = models["reward"].cuda(device)
    dynamics = models["dynamics"].cuda(device)

    # 2. Compress the first observation
    #obs_tensor = torch.Tensor(obs).cuda(device)
    obs_tensor = obs
    z_i = compression(obs_tensor)
    dim_b = z_i.shape[0]
    dim_z = z_i.shape[1]
    #action_seq_tensor = torch.Tensor(action_seq).cuda(device)
    action_seq_tensor = action_seq

    # 3. Set up storages
    r_mat = torch.Tensor(dim_b, horizon).cuda(device)
    z_mat = torch.Tensor(dim_b, horizon + 1, dim_z).cuda(device)
    z_mat[:,0,:] = z_i

    cumulative_reward = 0.0

    # 4. Rollout model 
    for t in range(horizon):
        u_t = action_seq_tensor[:,t,:]
        z_t = z_mat[:,t,:]

        zu_t = torch.cat((z_t, u_t), 1)

        r_mat[:,t] = reward(zu_t).squeeze(1)
        z_mat[:,t+1,:] = dynamics(zu_t)
        cumulative_reward += np.power(discount, t) * r_mat[:,t]

    # 5. Return results
    result = {
        "r_mat": r_mat,
        "z_mat": z_mat,
        "r_sum": cumulative_reward,
    }    

    return result 

def rollout_ensemble(obs, models, horizon, action_seq, discount=1.0, device="cuda:0"):
    # 1. Parse dictionary and setup models.
    compression = models["compression"].cuda(device)
    reward = models["reward"].cuda(device)
    dynamics = models["dynamics"].cuda(device)
    num_models = compression.num_models

    # 2. Compress the first observation
    # obs_tensor must be of dim B x 1 x 32 x 32.
    if not (type(obs) == type(torch.Tensor())):
        obs_tensor = torch.Tensor(obs)
    else:
        obs_tensor = obs

    if obs_tensor.device is not torch.device(device):
        obs_tensor = obs_tensor.cuda(device)
    # E x B x 1 x 32 x 32
    obs_tensor = torch.stack(num_models * [obs_tensor])

    # E x B x dimZ
    z_i = compression(obs_tensor)
    dim_z = compression.z
    dim_b = z_i.shape[1]

    # input is B x H x dimA.
    if not (type(action_seq) == type(torch.Tensor())):
        action_seq_tensor = torch.Tensor(action_seq)
    else:
        action_seq_tensor = action_seq 

    if action_seq_tensor.device is not torch.device(device):
        action_seq_tensor = action_seq_tensor.cuda(device)

    # change to E x B x H x dimA.
    action_seq_tensor = torch.stack(num_models * [action_seq_tensor])

    # 3. Set up storages
    r_mat = torch.Tensor(num_models, dim_b, horizon).cuda(device)
    z_mat = torch.Tensor(num_models, dim_b, horizon + 1, dim_z).cuda(device)
    z_mat[:,:,0,:] = z_i

    cumulative_reward = 0.0

    # 4. Rollout model 
    for t in range(horizon):
        u_t = action_seq_tensor[:,:,t,:]
        z_t = z_mat[:,:,t,:]

        zu_t = torch.cat((z_t, u_t), 2)

        r_mat[:,:,t] = reward(zu_t).squeeze(2)
        z_mat[:,:,t+1,:] = dynamics(zu_t)
        cumulative_reward += np.power(discount, t) * r_mat[:,:,t]

    # 5. Return results
    result = {
        "r_mat": r_mat,
        "z_mat": z_mat,
        "r_sum": cumulative_reward
    }    

    return result 


def Sampling_MPC(obs, models, horizon, num_candidates, action_lb, action_ub, discount=0.8, device="cuda"):
    """
    Sampling based MPC that does direct brute-force search.
    """
    # 1. Prase dictionary and setup models.
    compression = models["compression"].cuda(device)
    reward = models["reward"].cuda(device)
    dynamics = models["dynamics"].cuda(device)

    # 2. Sample a bunch of actions on horizon.
    dist = torch.distributions.uniform.Uniform(torch.Tensor(action_lb), torch.Tensor(action_ub))
    u = dist.sample([num_candidates, horizon]).cuda(device)

    obs_tensor = torch.Tensor(obs).cuda().unsqueeze(0) # 1 x H x W
    obs_tensor = torch.cat(num_candidates * [obs_tensor.unsqueeze(0)]) # B x 1 x H x W

    result = rollout_batch(obs_tensor, models, horizon, u, discount=discount)

    # argmin if cost, argmax if reward.
    best_index = torch.argmin(result["r_sum"])
    best_action = u[best_index, 0, :]

    result = {
        "action": best_action.detach().cpu().numpy(),
        "r_mat": result["r_mat"][best_index,:].detach().cpu().numpy(),
        "u_mat": u[best_index, :, :].detach().cpu().numpy()
    }

    return result

def Sampling_MPC_Ensemble(obs, models, horizon, num_candidates, action_lb, action_ub, discount=0.8, device="cuda"):
    """
    Sampling based MPC that does direct brute-force search.
    """
    # 1. Prase dictionary and setup models.
    compression = models["compression"].cuda(device)
    reward = models["reward"].cuda(device)
    dynamics = models["dynamics"].cuda(device)
    num_models = compression.num_models

    # 2. Sample a bunch of actions on horizon.
    dist = torch.distributions.uniform.Uniform(torch.Tensor(action_lb), torch.Tensor(action_ub))
    u = dist.sample([num_candidates, horizon]).cuda(device)

    obs_tensor = torch.Tensor(obs).cuda(device).unsqueeze(0) # 1 x H x W
    obs_tensor = torch.cat(num_candidates * [obs_tensor.unsqueeze(0)]) # B x 1 x H x W

    result = rollout_ensemble(obs_tensor, models, horizon, u, discount=discount, device=device)


    # argmin if cost, argmax if reward.
    best_index = torch.argmin(result["r_sum"][0,:])
    best_action = u[best_index, 0, :]

    result = {
        "action": best_action.detach().cpu().numpy(),
        "r_mat": result["r_mat"][0, best_index,:].detach().cpu().numpy(),
        "u_mat": u[best_index, :, :].detach().cpu().numpy()
    }

    return result    

def Sampling_MPC_Variance(obs, models, horizon, num_candidates, action_lb, action_ub, discount=0.8, device="cuda"):
    """
    Sampling based MPC that does direct brute-force search.
    """
    # 1. Prase dictionary and setup models.
    compression = models["compression"].cuda(device)
    reward = models["reward"].cuda(device)
    dynamics = models["dynamics"].cuda(device)
    num_models = compression.num_models

    # 2. Sample a bunch of actions on horizon.
    dist = torch.distributions.uniform.Uniform(torch.Tensor(action_lb), torch.Tensor(action_ub))
    u = dist.sample([num_candidates, horizon]).cuda(device)

    obs_tensor = torch.Tensor(obs).cuda(device).unsqueeze(0) # 1 x H x W
    obs_tensor = torch.cat(num_candidates * [obs_tensor.unsqueeze(0)]) # B x 1 x H x W

    result = rollout_ensemble(obs_tensor, models, horizon, u, discount=discount, device=device)

    # argmin if cost, argmax if reward.
    r_sum = result["r_sum"]
    r_sum_mean = torch.mean(r_sum, dim=0)
    r_sum_variance = torch.var(r_sum, dim=0)

    r_penalized = r_sum_mean + 0.0 * r_sum_variance

    best_index = torch.argmin(r_penalized)
    best_action = u[best_index, 0, :]

    result = {
        "action": best_action.detach().cpu().numpy(),
        "r_mat": result["r_mat"][:, best_index,:].detach().cpu().numpy(),
        "u_mat": u[best_index, :, :].detach().cpu().numpy()
    }

    return result        

def Sampling_CLF(obs, models, num_candidates, action_lb, action_ub, device="cuda"):
    """
    Sampling based CLF that imposes strict negativity of rewards.
    """
    # 1. Prase dictionary and setup models.
    compression = models["compression"].cuda(device)
    reward = models["reward"].cuda(device)
    dynamics = models["dynamics"].cuda(device)

    # 2. Sample a bunch of actions on horizon.
    dist = torch.distributions.uniform.Uniform(torch.Tensor(action_lb), torch.Tensor(action_ub))
    u = dist.sample([num_candidates, 2]).cuda(device)

    obs_tensor = torch.Tensor(obs).cuda().unsqueeze(0) # 1 x H x W
    obs_tensor = torch.cat(num_candidates * [obs_tensor.unsqueeze(0)]) # B x 1 x H x W

    result = rollout_batch(obs_tensor, models, 2, u)
    downhill_indices = torch.nonzero(result["r_mat"][:,0] >= result["r_mat"][:,1])

    # Choose a downhill index at random.
    random_ind = np.random.randint(downhill_indices.shape[1])
    best_index = downhill_indices[random_ind]

    best_action = u[best_index, 0, :]

    return best_action[0].detach().cpu().numpy()    


# TODO(terry-suh): refactor this to use the rollout function.
def OCO_MPC(obs, models, horizon, action_lb, action_ub, rate=1e-2, iterations=100, device="cuda"):
    """
    Simple Receding-Horizon MPC Based on Online Convex Optimization (Gradient Descent):
    @ params obs (np.array): observation at current time.
    @ params model (dict): dictionary of torch NNs with keys {"compression", "reward", "dynamics}.
    @ params horizon (int): number of horizonts to run the mpc for.
    @ params action_lb (np.array): lower bound on the actions given by a vector.
    @ params action_ub (np.array): upper bound on the actions given by a vector.
    @ params rate (float): rate of descent, also called learning rate. 
    @ params iterations (int): number of iterations to run gradient descent. 
    @ params device (string): device to run GD on. Supports "cuda", or "cpu".

    """
    # 1. Parse dictionary and setup models.
    compression = models["compression"].cuda(device)
    reward = models["reward"].cuda(device)
    dynamics = models["dynamics"].cuda(device)

    # 2. Set up decision variable and optimizer.
    action_lb = torch.Tensor(action_lb).cuda(device)
    action_ub = torch.Tensor(action_ub).cuda(device)

    dist = torch.distributions.uniform.Uniform(action_lb, action_ub)
    # Creates a (horizon x dim_action) matrix.
    u = dist.sample([horizon]).cuda(device)
    u.requires_grad = True
    optimizer = torch.optim.Adam([u], lr=rate)

    # 3. Implement saturation to comply with action bounds.
    def saturate(u):
        return torch.max(torch.min(u, action_ub), action_lb)

    # 4. Sum of cost to evaluate cost for an instance of u.
    z_i = compression(torch.Tensor(obs).cuda(device)).detach()
    # NOTE(terry-suh): Obtains dimension of z by assuming z_i will be (B x dim_z)
    # where B is batch. Verify this for your own implementation.
    dim_z = z_i.shape[1] 

    # 5. Run gradient descent with specified iterations.
    for _ in tqdm(range(iterations)):
        optimizer.zero_grad()

        # Do a forward pass to evaluate the cost.
        cost = torch.Tensor([[0.0]]).cuda(device)
        z_mat = torch.Tensor(horizon + 1, dim_z).cuda(device)
        z_mat[0,:] = z_i 
        for t in range(horizon):
            u_t = saturate(u[t,:])
            cost += reward(torch.cat((z_mat[t,:], u_t), 0).unsqueeze(0))
            z_mat[t+1, :] = dynamics(torch.cat((z_mat[t,:], u_t), 0).unsqueeze(0))

        # Backward pass and step.
        cost.backward()
        optimizer.step()

    # 6. Detach and return the first action.
    result = {
        "control": saturate(u[0,:]).detach().cpu().numpy(),
        "input_traj": u.detach().cpu().numpy(),
        "state_traj": z_mat.detach().cpu().numpy()
    }

    return result

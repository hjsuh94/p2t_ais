import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

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














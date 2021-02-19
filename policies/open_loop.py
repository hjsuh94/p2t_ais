import numpy as np
import torch

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
    dist = torch.distributions.uniform.Uniform(
        torch.Tensor(action_lb), torch.Tensor(action_ub))
    # Creates a (horizon x dim_action) matrix.
    u = dist.sample([horizon]).cuda(device).requires_grad()
    optimizer = torch.optim.Adam([u], lr=rate)

    # 3. Implement saturation to comply with action bounds.
    def saturate(u):
        return torch.max(torch.min(u, action_ub), action_lb)

    # 4. Sum of cost to evaluate cost for an instance of u.
    z_i = compression(torch.Tensor(obs).cuda(device))
    # NOTE(terry-suh): Obtains dimension of z by assuming z_i will be (B x dim_z)
    # where B is batch. Verify this for your own implementation.
    dim_z = z_i.shape[1] 

    def get_cost():
        cost = 0.0 
        # NOTE(terry-suh): Store Zs to avoid in-place operations on z which might
        # result in a messed-up computation graph? Check if this is unnecessary.
        z_mat = torch.Tensor(horizon + 1, dim_z)
        z_mat[0,:] = z_i
        for t in range(horizon):
            u_t = saturate(u[t,:])
            cost += reward(torch.cat((z_mat[t,:], u_t), 1))
            z_mat[t+1, :] = dynamics(torch.cat((z_mat[t,:], u_t), 1))
        return cost 

    # 5. Run gradient descent with specified iterations.
    for _ in range(iterations):
        optimizer.zero_grad()
        cost = get_cost()
        cost.backward()
        optimizer.step()

    # 6. Detach and return the first action.
    return saturate(u[0,:]).detach().cpu().numpy()













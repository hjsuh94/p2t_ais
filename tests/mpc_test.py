import numpy as np 
import torch
import torch.nn as nn
from policies.policy import OCO_MPC
import matplotlib.pyplot as plt


"""
MPC test. 
"""
class DoubleIntegratorCompression(nn.Module):
    def __init__(self):
        super(DoubleIntegratorCompression, self).__init__()

        self.Cinv = nn.Linear(2,2, bias=False)
        self.Cinv.weight.data = torch.eye(2)
        
    def forward(self, x): 
        # Assumes B x 2 observation.
        return self.Cinv(x)

class DoubleIntegratorDynamics(nn.Module):
    def __init__(self, dt):
        super(DoubleIntegratorDynamics, self).__init__()

        self.dt = dt

        self.A = nn.Linear(2,2, bias=False)
        self.A.weight.data = torch.Tensor([[1, dt], [0, 1]])

        self.B = nn.Linear(1,2, bias=False)
        self.B.weight.data = torch.Tensor([[0], [dt]])

    def forward(self, xu):
        # Assumes B x 3 xu, where B x 2 is state and B x 1 is input.
        x = xu[:,0:2]
        u = xu[:,2]
        return self.A(x) + self.B(u)

class DoubleIntegratorReward(nn.Module):
    def __init__(self):
        super(DoubleIntegratorReward, self).__init__()

        self.Q = nn.Linear(2, 2, bias=False)
        self.Q.weight.data = torch.eye(2)

        self.R = nn.Linear(1, 1, bias=False)
        self.R.weight.data = torch.eye(1)

    def forward(self, xu):
        x = xu[:,0:2]
        u = xu[:,2:3]
        return x @ self.Q(x).transpose(0, 1) + u @ self.R(u).transpose(0, 1)

models = {
    "compression": DoubleIntegratorCompression().train(),
    "dynamics": DoubleIntegratorDynamics(0.1).train(),
    "reward": DoubleIntegratorReward().train()
}


result = OCO_MPC(
    torch.Tensor([[1, -1]]), # Make sure this is a dim 2 tensor (B x 2)
    models = models,
    horizon=100,
    action_lb=torch.Tensor([-1]),
    action_ub=torch.Tensor([1])
)

plt.figure()
plt.plot(result["state_traj"][:,0], result["state_traj"][:,1])
plt.show()

plt.figure()
plt.plot(result["input_traj"])
plt.show()

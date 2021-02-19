import torch.nn as nn 
import torch 
import torch.nn.functional as F 
import numpy as np 
import os

"""
Models used for the AIS model in the onion problem. In the most general form,
there are 4 functions that are being attempted to learn:

1. r : z x a -> r (reward function from latent state to actual reward)
2. phi: z x a -> z (dynamics function from latent state and abstract action to next latent state)
3. sigma: y -> z (compression function from observation to latent state)
4. psi: u -> a (action abstraction function from input to action) 

We assume the following dimensions for input-output data. 

y \in R1024. 
u \in R4. 
r \in R1. 

The dimension of the (z,a) stands as a hyperparameter.
"""

"""
1. Reward function
"""
class RewardMLP(nn.Module):
    """
    1.1 Implements an arbitrary MLP to predict the reward function z x a -> r.
    """
    def __init__(self, z, a):
        super(RewardMLP, self).__init__()

        # Implements a 3 layer MLP. 
        # The number of weights in the intermediate layers are assigned somewhat arbitrarily,
        # but under the assumption that z + a ~ 100.
        self.reward_mlp = nn.Sequential(
            nn.Linear(z + a, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        return self.reward_mlp(x)

class RewardLinearNoAction(nn.Module):
    """
    1.2 Implements linear reward.
    """
    def __init__(self, z, a):
        super(RewardLinearNoAction, self).__init__()

        self.num_z = z
        self.R = nn.Linear(z, 1, bias=False)

    def forward(self, x):
        return self.R(x[:,0:self.num_z])


"""
2. Dynamics Function
"""
class DynamicsMLP(nn.Module):
    """
    2.1 Implements an arbitrary MLP for the dynamics function 
    """
    def __init__(self, z, a):
        super(DynamicsMLP, self).__init__()

        # Implements a 3 layer MLP for dynamics.
        self.dynamics_mlp = nn.Sequential(
            nn.Linear(z + a, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, z)
        )

    def forward(self, x):
        return self.dynamics_mlp(x)

class DynamicsLinear(nn.Module):
    """
    2.2 Implements a linear dynamics module.
    """
    def __init__(self, z, a):
        super(DynamicsLinear, self).__init__()

        # named AB since z = Az + Bu....
        self.AB = nn.Linear(z + a, z)

    def forward(self, x):
        return self.AB(x)

class DynamicsLinearNoAction(nn.Module):
    """
    2.3 Implements linear dynamics with no action.
    """
    def __init__(self, z, a):
        super(DynamicsLinearNoAction, self).__init__()

        self.A = nn.Linear(z, z, bias=False)
        self.num_z = z

    def forward(self, x):
        return self.A(x[:,0:self.num_z])

"""
3. Compression Function
"""
class CompressionMLP(nn.Module):
    """
    3.1 Implements a compression function I -> z.
    Note that this is a rather huge model because of the matrix size (though not unreasonably huge).
    """
    def __init__(self, z, a):
        super(CompressionMLP, self).__init__()

        # Implements a 3 layer MLP for compression.
        self.compression_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, z)
        )

    def forward(self, x):
        # Input is B x 1 x 32 x 32 image.
        b = x.shape[0]
        x = x.view(b, 1024)
        return self.compression_mlp(x)

class CompressionLinear(nn.Module):
    """
    3.2 Implements a linear compression function y->z.
    """
    def __init__(self, z, a):
        super(CompressionLinear, self).__init__()

        # named L since z = Ly in linear control.
        self.L = nn.Linear(1024, z)

    def forward(self, x):
        # Input is B x 1 x 32 x 32 image.
        b = x.shape[0]
        x = x.view(b, 1024)
        return self.L(x)

class CompressionConv(nn.Module):
    """
    3.3 Implements a compression function y -> z. Is a more efficient (translation-invariant)
    version of the CompressionMLP.
    """
    def __init__(self, z, a):
        super(CompressionConv, self).__init__()

        # Input image: 1 x 32 x 32
        self.compression_conv = nn.Sequential(
            nn.Conv2d(1, 32, 2, stride=2), # 32 x 16 x 16
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 2, stride=2), # 64 x 8 x 8 
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        # after compressing & flattening, pass it through another layer of mlps
        # to get the desired dimension of z.
        self.compression_conv_mlp = nn.Sequential(
            nn.Linear(64 * 8 * 8, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, z)
        )

    def forward(self, x):
        # input is B x 1 x 32 x 32 image.
        b = x.shape[0] # this is the batch size.
        x = self.compression_conv(x)
        x = x.view(b, 64 * 8 * 8)
        x = self.compression_conv_mlp(x)
        return x 

class CompressionLinearNoBias(nn.Module):
    """
    3.4 Linear with no bias
    """
    def __init__(self, z, a):
        super(CompressionLinearNoBias, self).__init__()

        self.L = nn.Linear(1024, z, bias=False)

    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, 1024)
        return self.L(x)        

"""
4. Action abstraction function
"""
class AbstractionMLP(nn.Module):
    """
    4.1 Implements the action abstraction function u -> z.
    """
    def __init__(self, z, a):
        super(AbstractionMLP, self).__init__()

        self.abstraction_mlp = nn.Sequential(
            nn.Linear(4, 100),
            nn.RelU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, a)
        )

    def forward(self, x):
        return self.abstraction_mlp(x)

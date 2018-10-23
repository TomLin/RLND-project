'''
The code is referred from
1. https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/tree/master/Chapter14
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class D4PGCritic(nn.Module):
    def __init__(self, obs_size, act_size, seed, n_atoms, v_min, v_max):
        super(D4PGCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(128 + act_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_atoms)
        )

        delta = (v_max - v_min) / (n_atoms - 1)
        self.register_buffer("supports", torch.arange(v_min, v_max + delta, delta))

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))

    def distr_to_q(self, distr):
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)


class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size, seed):
        super(DDPGActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, act_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


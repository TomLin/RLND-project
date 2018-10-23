'''
The code is referred from
1. https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/tree/master/Chapter14
'''


import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F

from d4pg_model import DDPGActor, D4PGCritic
from replay_memory import ReplayBuffer

GAMMA = 0.99
BATCH_SIZE = 64
LR_ACTOR = 1e-4
LR_CRITIC = 1e-4
BUFFER_SIZE = 100000
REWARD_STEPS = 1

LEARN_EVERY_STEP = 150
TAU = 1e-3   # for soft update of target parameters

Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def distr_projection(next_distr_v, rewards_v, dones_mask_t, gamma, device="cpu"):
    next_distr = next_distr_v.data.cpu().numpy()
    rewards = rewards_v.data.cpu().numpy()
    dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, N_ATOMS), dtype=np.float32)

    for atom in range(N_ATOMS):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * DELTA_Z) * gamma))
        b_j = (tz_j - Vmin) / DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

    if dones_mask.any():
        proj_distr[dones_mask] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones_mask]))
        b_j = (tz_j - Vmin) / DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones_mask.copy()
        eq_dones[dones_mask] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones_mask.copy()
        ne_dones[dones_mask] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return torch.FloatTensor(proj_distr).to(device)



class AgentD4PG():
    """
    Agent implementing noisy agent
    """
    def __init__(self, state_size, action_size, seed, device=device, epsilon=0.3):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.epsilon = epsilon

        self.t_step = 0 # counter for activating learning every few steps
        self.running_c_loss = 0
        self.running_a_loss = 0
        self.training_cnt = 0

        # Actor network (w/ target network)
        self.actor_local = DDPGActor(state_size, action_size, seed).to(device)
        self.actor_target = DDPGActor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic network (w/ target network)
        self.critic_local = D4PGCritic(state_size, action_size, seed, N_ATOMS, Vmin, Vmax).to(device)
        self.critic_target = D4PGCritic(state_size, action_size, seed, N_ATOMS, Vmin, Vmax).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def act(self, states, mode):
        states_v = torch.Tensor(np.array(states, dtype=np.float32)).to(self.device)

        self.actor_local.eval()
        with torch.no_grad():
            mu_v = self.actor_local(states_v)
            actions = mu_v.data.cpu().numpy()
        self.actor_local.train()

        if mode == "test":
            return np.clip(actions, -1, 1)

        elif mode == "train":
            actions += self.epsilon * np.random.normal(size=actions.shape)
            actions = np.clip(actions, -1, 1)
            return actions

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # activate learning every few steps
        self.t_step = self.t_step + 1
        if self.t_step % LEARN_EVERY_STEP == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
                for _ in range(10):  # update 10 times per learning
                    experiences = self.memory.sample2()
                    self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        crt_distr_v = self.critic_local(states, actions)
        last_act_v = self.actor_target(next_states)
        last_distr_v = F.softmax(self.critic_target(next_states, last_act_v), dim=1)

        proj_distr_v = distr_projection(last_distr_v, rewards, dones,
                                        gamma=gamma ** REWARD_STEPS, device=device)

        prob_dist_v = -F.log_softmax(crt_distr_v, dim=1) * proj_distr_v
        critic_loss_v = prob_dist_v.sum(dim=1).mean()

        self.running_c_loss += float(critic_loss_v.cpu().data.numpy())
        self.training_cnt += 1

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss_v.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        crt_distr_v  = self.critic_local(states, actions_pred)
        actor_loss_v = -self.critic_local.distr_to_q(crt_distr_v)
        actor_loss_v = actor_loss_v.mean()
        self.running_a_loss += float(actor_loss_v.cpu().data.numpy())

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss_v.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)  # clip gradient to max 1
        self.actor_optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)





import random
import numpy as np
import torch
import torch.optim as optim
from ddpg_model import Actor, Critic
from .prioritized_memory import PrioritizedMemory
from noise import OUNoise

LR_ACTOR = 1e-5
LR_CRITIC = 1e-4
WEIGHT_DECAY = 1e-2 # L2 weight decay
BATCH_SIZE = 64
BUFFER_SIZE = 50000
LEARN_EVERY_STEP = 20
GAMMA = 0.99 # discount factor
TAU = 1e-3              # for soft update of target parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    '''Interact with and learn from environment.'''

    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.t_step = 0 # counter for activating learning every few steps
        self.running_c_loss = 0
        self.running_a_loss = 0
        self.training_cnt = 0

        # Actor network (w/ target network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic network (w/ target network)
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, seed)

        # Prioritized replay memory
        self.prioritized_memory = PrioritizedMemory(BATCH_SIZE, BUFFER_SIZE, seed)

    def act(self, state, mode):
        '''Returns actions for given state as per current policy.

        Params
        ======
            state (array): current state
            mode (string): train or test
            epsilon (float): for epsilon-greedy action selection
        '''
        state = torch.from_numpy(state).unsqueeze(0).float().to(device) # shape of state (1, state_size)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if mode == 'test':
            return np.clip(action, -1, 1)

        elif mode == 'train': # if train, then add OUNoise in action
            action += self.noise.sample()
            return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        # add new experience in memory
        self.prioritized_memory.add(state, action, reward, next_state, done)

        # activate learning every few steps
        self.t_step = self.t_step + 1
        if self.t_step % LEARN_EVERY_STEP == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.prioritized_memory) >= BUFFER_SIZE:
                for _ in range(10): # update 10 times per learning
                    idxes, experiences, is_weights = self.prioritized_memory.sample(device)
                    self.learn(experiences, GAMMA, is_weights=is_weights, leaf_idxes=idxes)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, is_weights, leaf_idxes):
        """
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Policy loss = (1/n)*Q_local(s,a) -> for deterministic policy (no log prob)

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            is_weights (tensor array): importance-sampling weights for prioritized experience replay
            leaf_idxes (numpy array): indexes for update priorities in SumTree
        """

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        rewards = rewards # TODO: rewards are clipped to be in [-1,1]

        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss

        Q_expected = self.critic_local(states, actions)
        td_errors = (Q_targets - Q_expected).tanh()  # TD-errors are clipped to be in [-1,1]
        abs_errors = td_errors.abs().cpu().data.numpy()  # pull back to cpu
        self.prioritized_memory.batch_update(leaf_idxes, abs_errors)  # update priorities in SumTree

        c_loss = (is_weights * (td_errors ** 2)).mean()  # adjust squared TD loss by Importance-Sampling Weights
        self.running_c_loss += float(c_loss.cpu().data.numpy())
        self.training_cnt += 1

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        c_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) # clip gradient to max 1
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        a_loss = self.critic_local(states, actions_pred)
        a_loss = -a_loss.mean()
        self.running_a_loss += float(a_loss.cpu().data.numpy())

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)  # clip gradient to max 1
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







'''
The code is referred from
1. https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum

'''


from collections import deque, namedtuple
import random
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.seed = random.seed(seed)

    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory.

        :param
            states (n_agents, state_size) (numpy)
            actions (n_agents, action_size) (numpy)
            rewards (n_agents,) (numpy)
            next_states (n_agents, state_size) (numpy)
            dones (n_agents,) (numpy)
        """
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory.

        :return
            b_a_states (batch_size, n_agents, state_size) (tensor)
            b_a_actions (batch_size, n_agents, action_size) (tensor)
            b_rewards (batch_size, n_agents) (numpy)
            b_a_next_states (batch_size, n_agents, next_states) (tensor)
            b_dones (batch_size, n_agents) (tensor)

        """
        experiences = random.sample(self.memory, k=self.batch_size)

        b_a_states = torch.from_numpy(np.vstack([np.expand_dims(e.states, axis=0) for e in experiences if e is not None])).float().to(device)
        b_a_actions = torch.from_numpy(np.vstack([np.expand_dims(e.actions, axis=0) for e in experiences if e is not None])).float().to(device)
        b_rewards = torch.from_numpy(np.vstack([np.expand_dims(e.rewards, axis=0) for e in experiences if e is not None])).float().to(device)
        b_a_next_states = torch.from_numpy(np.vstack([np.expand_dims(e.next_states, axis=0) for e in experiences if e is not None])).float().to(
            device)
        b_dones = torch.from_numpy(np.vstack([np.expand_dims(e.dones, axis=0) for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return b_a_states, b_a_actions, b_rewards, b_a_next_states, b_dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

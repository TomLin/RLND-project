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
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def sample2(self, device=device):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for exp in experiences:
            states.append(exp.state.squeeze(0))
            actions.append(exp.action.squeeze(0))
            rewards.append(exp.reward)
            dones.append(exp.done)
            next_states.append(exp.next_state.squeeze(0))

        states_v = torch.Tensor(np.array(states, dtype=np.float32)).to(device)
        actions_v = torch.Tensor(np.array(actions, dtype=np.float32)).to(device)
        rewards_v = torch.Tensor(np.array(rewards, dtype=np.float32)).to(device)
        next_states_v = torch.Tensor(np.array(next_states, dtype=np.float32)).to(device)
        dones_v = torch.ByteTensor(dones).to(device)

        return states_v, actions_v, rewards_v, next_states_v, dones_v
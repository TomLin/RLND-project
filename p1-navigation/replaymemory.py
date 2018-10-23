import numpy as np 
import random
from collections import deque, namedtuple
import torch

class ReplayMemory:
    '''ReplayMemory with uniformly distributed probability on samples.'''

    def __init__(self, batch_size, buffer_size, seed):
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.experience = namedtuple('experience',
            field_names=['state','action','reward','next_state','done'])
        self.memory = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        ''' Add new experience in memory.

        Params
        ======
            state(list),
            action(int),
            reward(float),
            next_state(list),
            done(boolean)
        '''
        new_e = self.experience(state, action, reward, next_state, done)
        self.memory.append(new_e)

    def sample(self, device):
        '''Sample a batch of experiences from memory.
        
        Params
        ======
            device(string): either 'cpu' or 'cuda:0'
        '''
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
            ).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        '''Return the length of current memory.'''
        return len(self.memory)
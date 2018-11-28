'''
The code is referred from
1. https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum

'''


import random
import copy
import numpy as np

class OUNoise():
    '''Ornstein-Uhlenbeck process.'''

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        '''Initialize parameters and noise process.'''
        self.mu = mu * np.ones(size) # action size
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        '''Reset the internal state (= noise) to mean (mu).'''
        self.state = copy.copy(self.mu)

    def sample(self):
        '''Update internal state and return it as a noise sample.'''
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state
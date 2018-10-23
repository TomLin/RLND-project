from sumtree import SumTree
import numpy as np
import random
import torch

class PrioritizedMemory(object):
    '''Memory object with proportional prioritized probability on samples.

    The code is modified from
    1. https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN
    2. https://github.com/rlcode/per
    
    '''

    def __init__(self, batch_size, buffer_size, seed):
        self.seed = random.seed(seed)
        self.epsilon = 0.01 # small amount to avoid zero priority
        self.alpha = 0.6 # [0~1] convert the importance of TD error to priority,
                         # it is a trade-off between using priority and totally uniformly randomness
        # self.absolute_error_upper = 1.0 # clipped abs error (abs error is the absolute value of TD error)
        self.beta = 0.4 # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = 0.001
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.sumtree = SumTree(buffer_size)
        

    def __len__(self):
        '''Track how many experiences being added in the memory'''
        not_none = np.array([1 for i in self.sumtree.memory if i is not None])
        count = np.sum(not_none)
        return count
    
    def add(self, state, action, reward, next_state, done):
        '''Add new experience meanwhile assign it the maximum priority from the current group.'''

        # assign maximal priority from current group for new experience
        max_priority = np.max(self.sumtree.tree[-self.buffer_size:])

        if max_priority == 0:
            max_priority = 1.
        
        self.sumtree.store(max_priority, state, action, reward, next_state, done) # set the max_priority for new experience

    def sample(self, device):
        '''Sample a batch of examples. 
        First, divide the range (0, total_priority) into batch_size segments.
        Then uniformly sample one value from each segment. 
        Use the value in get_leaf() to search through the tree and retrieve the closest associated leaf_idx.
        In addition, compute the relevant is_weights (Importance-Sampling Weights).

        Params
        ======
            device(string): either 'cpu' or 'cuda:0'

        Return
        ======
            idxes (array): leaf index for SumTree
            (states, actions, rewards, next_states, dones) (tensor tuple): sampled experiences in tuple
            is_weights (tensor array): importance-sampling weights for experiences
        '''

        idxes = []
        experiences = []
        priorities = []
        n_segments = self.sumtree.total_priority/self.batch_size 
        
        # increase beta each time when sampling a new minibatch
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling]) # max = 1

        for i in range(self.batch_size):
            a, b = n_segments * i, n_segments * (i + 1)
            value = random.uniform(a,b)
            leaf_idx, priority, experience_tuple = self.sumtree.get_leaf(value)
            if experience_tuple is not None:
                priorities.append(priority)
                experiences.append(experience_tuple)
                idxes.append(leaf_idx)
        
        sampling_probabilities = np.array(priorities) / self.sumtree.total_priority
        # compute the max is_weights for the batch
        is_weights = np.power(self.sumtree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        is_weights = is_weights.reshape((-1,1))
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences]).astype(np.uint8)
            ).float().to(device)        

        is_weights = torch.from_numpy(is_weights).float().to(device)
        idxes = np.array(idxes)

        return idxes, (states, actions, rewards, next_states, dones), is_weights
    
    def batch_update(self, leaf_idxes, abs_errors):
        '''Update the sample batch's priorities and their parent node's priorities.
        
        Params
        ======
            leaf_idxes (numpy array): leaf indexes for SumTree
            abs_errors (numpy array): the absolute value of TD error

        '''

        abs_errors += self.epsilon # convert to abs and avoid 0
        priorities = np.power(abs_errors, self.alpha)
        for idx, p in zip(leaf_idxes, priorities):
            self.sumtree.update(idx, p)
    
    

    

    

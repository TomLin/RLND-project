from collections import namedtuple
import numpy as np

class SumTree:
    '''Store experience in the memory and its priority in the tree.
    
    The code is modified from
    1. https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN
    2. https://github.com/rlcode/per 
    '''

    def __init__(self, buffer_size):
        self.memory_idx = 0
        self.n_entries = 0
        self.buffer_size = buffer_size
        self.tree = np.zeros(2*self.buffer_size - 1)
        self.experience = namedtuple('experience',
            ['state','action','reward','next_state','done']) # initialize namedtuple class
        self.memory = [None] * buffer_size

    def update(self, leaf_idx, priority):
        '''Update the priority of the leaf_idx and also propagate the priority-change through tree.
        
        Params
        ======
            leaf_idx (int)
            priority (float)
        '''

        priority_change = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority

        # then propagate the priority change through tree
        tree_idx = leaf_idx
        while tree_idx != 0:
            tree_idx = (tree_idx-1)//2
            self.tree[tree_idx] += priority_change
    
    def store(self, priority, state, action, reward, next_state, done):
        '''Store new experience in memory and update the relevant priorities in tree. 
        The new experience will overwrite the old experience from the beginning once the memory is full.

        Params
        ======
            priority (float)
            state (array)
            action (int)
            reward (float)
            next_state (array)
            done (boolean)
        '''

        leaf_idx  = self.memory_idx + self.buffer_size - 1
        new_e = self.experience(state, action, reward, next_state, done)        
        self.memory[self.memory_idx] = new_e # update experience
        self.update(leaf_idx, priority) # update priorities in tree

        self.memory_idx += 1
        if self.memory_idx >= self.buffer_size: # replace the old experience when exceeding buffer_size
            self.memory_idx = 0
        
        if self.n_entries < self.buffer_size:
            self.n_entries += 1

    def get_leaf(self, value):
        '''Use the value to search through the tree and 
        retrieve the closest associated leaf_idx and its memory.
        
        Params
        ======
            value (float): used to search through the tree for closest leaf_idx

        Return
        ======
            leaf_idx (int)
            priority (float)
            experience (namedtuple)

        '''

        parent_idx = 0
        while True: # this node's left and right kids
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # If we reach bottom, end the search
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else: # downward search, always search for a higher priority node
                if value <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx                    
                else:
                    value -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        
        memory_idx = leaf_idx - self.buffer_size + 1
        return leaf_idx, self.tree[leaf_idx], self.memory[memory_idx]

    @property
    def total_priority(self):
        return self.tree[0] # the total priorities stored in the root node


    
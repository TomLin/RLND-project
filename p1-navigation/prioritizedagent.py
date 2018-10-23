
import random
import numpy as np
from model import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim
from prioritizedmemory import PrioritizedMemory

LR = 0.0000625 # learning rate: one-forth of normal agent's LR 
BATCH_SIZE = 32
BUFFER_SIZE = 50000
GAMMA = 0.99 # discount factor
LEARN_EVERY_STEP = 1
UPDATE_EVERY_STEP = 500 # update target network

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

class PrioritizedAgent:
    '''Interact with and learn from the environment.
    The agent uses prioritized experience replay.
    '''

    def __init__(self, state_size, action_size, seed, is_double_q=False):
        '''Initialize an Agent.

        Params
        ======
            state_size (int): the dimension of the state
            action_size (int): the number of actions
            seed (int): random seed
        '''

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.t_step = 0 # Initialize time step (for tracking LEARN_EVERY_STEP and UPDATE_EVERY_STEP)
        self.running_loss = 0
        self.training_cnt = 0

        self.is_double_q = is_double_q

        self.qnetwork_local = QNetwork(self.state_size, self.action_size, seed).to(device)
        self.qnetowrk_target = QNetwork(self.state_size, self.action_size, seed).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.prioritized_memory = PrioritizedMemory(BATCH_SIZE, BUFFER_SIZE, seed)

    def act(self, state, mode, epsilon=None):
        '''Returns actions for given state as per current policy.
        
        Params
        ======
            state (array): current state
            mode (string): train or test
            epsilon (float): for epsilon-greedy action selection
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device) # shape of state (1, state)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local.forward(state)
        self.qnetwork_local.train()

        if mode == 'test':
            action = np.argmax(action_values.cpu().data.numpy()) # pull action values from gpu to local cpu
        
        elif mode == 'train':
            if random.random() <= epsilon: # random action
                action = random.choice(np.arange(self.action_size))
            else: # greedy action
                action = np.argmax(action_values.cpu().data.numpy()) # pull action values from gpu to local cpu
        
        return action
    
    def step(self, state, action, reward, next_state, done):
        # add new experience in memory
        self.prioritized_memory.add(state, action, reward, next_state, done)

        # activate learning every few steps
        self.t_step = self.t_step + 1
        if self.t_step % LEARN_EVERY_STEP == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.prioritized_memory) >= BUFFER_SIZE:
                idxes, experiences, is_weights = self.prioritized_memory.sample(device)
                self.learn(experiences, GAMMA, is_weights=is_weights, leaf_idxes=idxes)

    def learn(self, experiences, gamma, is_weights, leaf_idxes):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            is_weights (tensor array): importance-sampling weights for prioritized experience replay
            leaf_idxes (numpy array): indexes for update priorities in SumTree

        """
        
        states, actions, rewards, next_states, dones = experiences
        
        q_local_chosen_action_values = self.qnetwork_local.forward(states).gather(1, actions)
        q_target_action_values = self.qnetowrk_target.forward(next_states).detach()

        if self.is_double_q == True:
            q_local_next_actions = self.qnetwork_local.forward(next_states).detach().max(1)[1].unsqueeze(1) # shape (batch_size, 1)
            q_target_best_action_values = q_target_action_values.gather(1, q_local_next_actions) # Double DQN
        
        elif self.is_double_q == False:
            q_target_best_action_values = q_target_action_values.max(1)[0].unsqueeze(1) # shape (batch_size, 1)
        
        rewards = rewards.tanh() # rewards are clipped to be in [-1,1], referencing from original paper
        q_target_values = rewards + gamma * q_target_best_action_values * (1 - dones) # zero value for terminal state

        td_errors = (q_target_values - q_local_chosen_action_values).tanh() # TD-errors are clipped to be in [-1,1], referencing from original paper
        abs_errors = td_errors.abs().cpu().data.numpy() # pull back to cpu
        self.prioritized_memory.batch_update(leaf_idxes, abs_errors) # update priorities in SumTree

        loss = (is_weights * (td_errors**2)).mean() # adjust squared TD loss by Importance-Sampling Weights

        self.running_loss += float(loss.cpu().data.numpy())
        self.training_cnt += 1

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        if self.t_step % UPDATE_EVERY_STEP == 0:
            self.update(self.qnetwork_local, self.qnetowrk_target)

    def update(self, local_netowrk, target_network):
        """Hard update model parameters, as indicated in original paper.
        
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        for local_param, target_param in zip(local_netowrk.parameters(), target_network.parameters()):
            target_param.data.copy_(local_param.data)


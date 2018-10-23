import numpy as np 
import random
from model import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim
from replaymemory import ReplayMemory

BUFFER_SIZE = 50000    # replay buffer size
BATCH_SIZE = 32         # minibatch size
GAMMA = 0.99            # discount factor
LR = 0.00025             # learning rate 
LEARN_EVERY_STEP = 1    # how often to activate the learning process
UPDATE_EVERY_STEP = 500 # how often to update the target network parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

class Agent:
    '''Interact with and learn from the environment.'''

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
        self.replay_memory = ReplayMemory(BATCH_SIZE, BUFFER_SIZE, seed)

    def act(self, state, mode, epsilon=None):
        '''Returns actions for given state as per current policy.
        
        Params
        ======
            state (array-like): current state
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
        self.replay_memory.add(state, action, reward, next_state, done)

        # activate learning every few steps
        self.t_step = self.t_step + 1
        if self.t_step % LEARN_EVERY_STEP == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.replay_memory) >= BUFFER_SIZE:
                experiences = self.replay_memory.sample(device)
                self.learn(experiences, GAMMA)
        
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor

        """

        # compute and minimize the loss        
        states, actions, rewards, next_states, dones = experiences

        q_local_chosen_action_values = self.qnetwork_local.forward(states).gather(1, actions)
        q_target_action_values = self.qnetowrk_target.forward(next_states).detach() # # detach from graph, don't backpropagate
        
        if self.is_double_q == True:
            q_local_next_actions = self.qnetwork_local.forward(next_states).detach().max(1)[1].unsqueeze(1) # shape (batch_size, 1)
            q_target_best_action_values = q_target_action_values.gather(1, q_local_next_actions) # Double DQN

        elif self.is_double_q == False:  
            q_target_best_action_values = q_target_action_values.max(1)[0].unsqueeze(1) # shape (batch_size, 1)
                
        q_target_values = rewards + gamma * q_target_best_action_values * (1 - dones) # zero value for terminal state 
        
        td_errors = q_target_values - q_local_chosen_action_values

        loss = (td_errors**2).mean()

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
        
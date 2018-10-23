'''
The code is referred from
1. https://github.com/MorvanZhou/pytorch-A3C/blob/master/continuous_A3C.py
2. https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/network/network_heads.py
'''




import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 



N_INPUTS = 33
N_ACTIONS = 4
GAMMA = 0.99 # reward discount rate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class A2CModel(nn.Module):
    def __init__(self):
        super(A2CModel, self).__init__()
        self.fc1 = nn.Linear(N_INPUTS, 128)
        self.fc2 = nn.Linear(128, 64)
        self.actor = nn.Linear(64, N_ACTIONS)
        self.critic = nn.Linear(64, 1)
        self.std = torch.ones(N_ACTIONS).to(device)
        self.dist = torch.distributions.Normal

    def forward(self, s):
        '''
        Params
        ======
            s (n_process, state_size) (tensor): states

        '''
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))

        return s
    
    def get_action(self, s):
        '''
        Params
        ======
            s (n_process, state_size) (tensor): states

        Returns
        ======
            action_tanh (n_process, action_size) (tensor): action limited within (-1,1)
            action (n_process, action_size) (tensor): raw action
        '''
        s = self.forward(s)
        mu = self.actor(s)
        dist_ = self.dist(mu, self.std)
        action = dist_.sample()
        action_tanh = F.tanh(action)
        return action_tanh, action
    
    def get_action_prob(self, s, a):
        '''
        Params
        ======
            s (n_process, state_size) (tensor): states
            a (n_process, action_size) (tensor): actions
        
        Returns
        =======
            mu (n_process, action_size) (tensor): mean value of action distribution
            self.std (action_size,) (tensor): the standard deviation of every action
            log_prob (n_process,) (tensor): log probability of input action
        '''
        
        s = self.forward(s)
        mu = self.actor(s)
        dist_ = self.dist(mu, self.std)
        log_prob  = dist_.log_prob(a)
        log_prob = torch.sum(log_prob, dim=1, keepdim=False)
        return mu, self.std, log_prob
        
    def get_state_value(self, s):
        '''
        Params
        ======
            s (n_process, state_size) (tensor): states

        Returns
        =======
            value (n_process,) (tensor)
        '''
        s = self.forward(s)
        value = self.critic(s).squeeze(1)
        return value

def collect_trajectories(model, env, brain_name, init_states, episode_end, n_steps):
    '''
    Params
    ======
        model (object): A2C model
        env (object): environment
        brain_name (string): brain name of environment
        init_states (n_process, state_size) (numpy): initial states for loop
        episode_end (bool): tracker of episode end, default False
        n_steps (int): number of steps for reward collection
    Returns
    =======
        batch_s (T, n_process, state_size) (numpy): batch of states
        batch_a (T, n_process, action_size) (numpy): batch of actions
        batch_v_t (T, n_process) (numpy): batch of n-step rewards (aks target value)
        accu_rewards (n_process,) (numpy): accumulated rewards for process (being summed up on all process)
        init_states (n_process, state_size) (numpy): initial states for next batch
        episode_end (bool): tracker of episode end
    '''

    batch_s = []
    batch_a = []
    batch_r = []

    states = init_states
    accu_rewards = np.zeros(init_states.shape[0])

    t = 0
    while True:
        t += 1

        model.eval()
        with torch.no_grad():
            states = torch.from_numpy(states).float().to(device)
            actions_tanh, actions = model.get_action(states)
        model.train()
        # actions_tanh (n_process, action_size) (tensor), actions limited within (-1,1)
        # actions (n_process, action_size) (tensor)
        
        env_info = env.step(actions_tanh.cpu().data.numpy())[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        # next_states (numpy array)
        # rewards (list)
        # dones (list)
        rewards = np.array(rewards)
        dones = np.array(dones)
        
        accu_rewards += rewards

        batch_s.append(states.cpu().data.numpy()) # final shape of batch_s (T, n_process, state_size) (list of numpy)
        batch_a.append(actions.cpu().data.numpy()) # final shape of batch_a (T, n_process, action_size) (list of numpy)
        batch_r.append(rewards) # final shape of batch_r (T, n_process) (list of numpy array)

        if dones.any() or t >= n_steps:
            model.eval()
            next_states = torch.from_numpy(next_states).float().to(device)
            final_r = model.get_state_value(next_states).detach().cpu().data.numpy() # final_r (n_process,) (numpy)
            model.train()

            for i in range(len(dones)):
                if dones[i] == True:
                    final_r[i] = 0
                else:
                    final_r[i] = final_r[i]

            batch_v_t = [] # compute n-step rewards (aks target value)
            batch_r = np.array(batch_r)
            
            for r in batch_r[::-1]:
                mean = np.mean(r)
                std = np.std(r)
                r = (r - mean)/(std+0.0001) # normalize rewards in n_process on each timestep
                final_r = r + GAMMA * final_r
                batch_v_t.append(final_r)
            batch_v_t = np.array(batch_v_t)[::-1] # final shape (T, n_process) (numpy)

            break

        states = next_states

    if dones.any():
        env_info = env.reset(train_mode=True)[brain_name]
        init_states = env_info.vector_observations
        episode_end = True
        
    else:
        init_states = next_states.cpu().data.numpy() # if not done, continue batch collection from last states

    batch_s = np.stack(batch_s)
    batch_a = np.stack(batch_a)

    return batch_s, batch_a, batch_v_t, np.sum(accu_rewards), init_states, episode_end


def learn(batch_s, batch_a, batch_v_t, model, optimizer):
    '''
    Params
    ======
        batch_s (T, n_process, state_size) (numpy)
        batch_a (T, n_process, action_size) (numpy): batch of actions
        batch_v_t (T, n_process) (numpy): batch of n-step rewards (aks target value)
        model (object): A2C model
        optimizer (object): model parameter optimizer

    Returns
    ======
        total_loss (int): mean actor-critic loss for each batch 


    '''

    batch_s_ = torch.from_numpy(batch_s).float().to(device)
    batch_s_ = batch_s_.view(-1, batch_s.shape[-1]) # shape from (T,n_process,state_size) -> (TxN, state_size)

    batch_a_ = torch.from_numpy(batch_a).float().to(device)
    batch_a_ = batch_a_.view(-1, batch_a.shape[-1]) # shape from (T,n_process,action_size) -> (TxN, action_size)

    values = model.get_state_value(batch_s_) # shape (TxN,)
    values = values.view(*batch_s.shape[:2]) # shape (T,n)

    # pytorch's problem of negative stride -> require .copy() to create new numpy
    batch_v_t_ = torch.from_numpy(batch_v_t.copy()).float().to(device)
    td = batch_v_t_ - values # shape (T, n_process) (tensor)
    c_loss = td.pow(2).mean()

    mus, stds, log_probs = model.get_action_prob(batch_s_, batch_a_)
    log_probs_ = log_probs.view(*batch_a.shape[:2]) # shape from (TxN,) -> (T,n) (tensor)

    a_loss = -((log_probs_ * td.detach()).mean())
    total_loss = c_loss + a_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # stds is constnat -> no gradient, no detach()
    return total_loss.detach().cpu().data.numpy(), mus.detach().cpu().data.numpy(), stds.cpu().data.numpy()

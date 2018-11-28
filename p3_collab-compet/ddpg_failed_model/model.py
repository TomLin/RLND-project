import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def hidden_init(layer):
	'''
	Provide fan in (the number of input units) of each hidden layer
	as the component of normalizer.

	:param
		layer: hidden layer

	:return
		(-lim, lim): tuple of min and max value for uniform distribution
	'''

	fan_in = layer.weight.data.size()[0]
	lim = 1. / np.sqrt(fan_in)
	return (-lim, lim)


class Actor(nn.Module):

	def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
		'''
		Initialize parameters and build model.

		:param
			state_size (int)
			action_size (int)
			seed (int)
			fc1_units (int)
			fc2_units (int)
		'''

		super(Actor, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.fc1 = nn.Linear(state_size, fc1_units)
		self.fc2 = nn.Linear(fc1_units, fc2_units)
		self.fc3 = nn.Linear(fc2_units, action_size)
		self.reset_parameters()

	def reset_parameters(self):
		self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
		self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
		self.fc3.weight.data.uniform_(-3e-3, 3e-3)

	def forward(self, state):
		'''Build an actor (policy) network that maps state to actions

		:param:
			state (batch_size, state_size) (tensor): partial observation of current agent

		:return:
			actions (batch_size, action_size) (tensor): being limited within range (-1,1)
		'''

		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		return F.tanh(self.fc3(x))


class Critic(nn.Module):

	def __init__(self, state_size, action_size, n_agent, seed, fc1_units=64, fc2_units=64):
		super(Critic, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.n_agent = n_agent
		self.state_size = state_size # observation of single agent
		self.action_size = action_size # action of single agent

		self.all_obs_size = state_size * n_agent # observations of all agents
		self.all_act_size = action_size * n_agent # actions of all agents

		self.fc1 = nn.Linear(self.all_obs_size, fc1_units)
		self.fc2 = nn.Linear(fc1_units + self.all_act_size, fc2_units)  # actions are included in latter phase
		self.fc3 = nn.Linear(fc2_units, 1)
		self.reset_parameters()

	def reset_parameters(self):
		self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
		self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
		self.fc3.weight.data.uniform_(-3e-3, 3e-3)

	def forward(self, all_obs, all_act):
		'''
		Build a critic (value) network that maps (state, action) pairs to Q-value.

		:param
			all_obs (batch_size, all_obs_size) (tensor)
			all_act (batch_size, all_act_size) (tensor)

		:return
			q_value (batch_size, 1) (tensor)
		'''

		s1 = F.relu(self.fc1(all_obs))
		sa1 = torch.cat((s1, all_act), dim=1)
		x = F.relu(self.fc2(sa1))
		return self.fc3(x)

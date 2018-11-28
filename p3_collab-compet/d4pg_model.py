'''
The code is partially referred from 1. https://github.com/kelvin84hk/DRLND_P3_collab-compet

'''



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
		super(Actor, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.fc1 = nn.Linear(state_size, fc1_units)
		self.fc2 = nn.Linear(fc1_units, fc2_units)
		self.fc3 = nn.Linear(fc2_units, action_size)
		self.b3 = nn.BatchNorm1d(action_size)
		self.tanh = nn.Tanh()
		self.reset_parameters()

	def reset_parameters(self):
		self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
		self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
		self.fc3.weight.data.uniform_(-3e-3, 3e-3)

	def forward(self, state):
		x = F.leaky_relu(self.fc1(state))
		x = F.leaky_relu(self.fc2(x))
		x = F.leaky_relu(self.fc3(x))
		x = self.b3(x)
		x = self.tanh(x)

		return x

class CriticD4PG(nn.Module):

	def __init__(self, state_size, action_size, seed, n_atoms, v_max, v_min, fc1_units=64, fc2_units=64):
		super(CriticD4PG, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.fc1 = nn.Linear(state_size, fc1_units)
		self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
		self.fc3 = nn.Linear(fc2_units, n_atoms)
		delta = (v_max - v_min) / (n_atoms - 1)
		self.register_buffer("supports", torch.arange(v_min, v_max + delta, delta))
		self.reset_parameters()

	def reset_parameters(self):
		self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
		self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
		self.fc3.weight.data.uniform_(-3e-3, 3e-3)

	def forward(self, state, action):
		# xs=torch.cat((state,action_oppo), dim=1)
		xs = F.leaky_relu(self.fc1(state))
		x = torch.cat((xs, action), dim=1)
		x = F.leaky_relu(self.fc2(x))
		x = self.fc3(x)

		return x

	def distr_to_q(self, distr):
		weights = F.softmax(distr, dim=1) * self.supports
		res = weights.sum(dim=1)
		return res.unsqueeze(dim=-1)




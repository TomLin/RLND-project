from collections import deque, namedtuple
import random
import numpy as np
import torch

Experience = namedtuple('Experience', ('states', 'actions', 'rewards', 'next_states', 'dones'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayMemory:

	def __init__(self, capacity, batch_size):
		self.capacity = capacity
		self.batch_size = batch_size
		self.memory = deque(maxlen=capacity)
		self.position = 0

	def add(self, *args):
		self.memory.append(Experience(*args))

	def sample(self):
		"""

		:return:
			b_a_states (batch_size, n_agents, state_size) (tensor): batch of n agents' state for critic update
			b_a_actions (batch_size, n_agents, action_size) (tensor): batch of n agents' action for critic update
			b_a_next_states (batch_size, n_agents, state_size) (tensor): batch of n agents' next state for critic update
			b_rewards (batch_size, n_agents) (tensor): batch of n agents' rewards for critic update
			b_dones (batch_size, n_agents) (tensor): batch of n agents' dones
		"""

		experiences = random.sample(self.memory, self.batch_size)

		# b_a_states: (batch_size, n_agents, state_size) batch of n_agent's state for critic update
		b_a_states = torch.from_numpy(
			np.vstack([np.expand_dims(e.states, axis=0) for e in experiences if e is not None])
			).float().to(device)

		# b_a_actions: (batch_size, n_agents, action_size) batch of n_agent's action for critic update
		b_a_actions = torch.from_numpy(
			np.vstack([np.expand_dims(e.actions, axis=0) for e in experiences if e is not None])
			).float().to(device)

		# b_a_next_states: (batch_size, n_agents, state_size) batch of n_agent's next state for critic update
		b_a_next_states = torch.from_numpy(
			np.vstack([np.expand_dims(e.next_states, axis=0) for e in experiences if e is not None])
			).float().to(device)

		# b_rewards: (batch_size, n_agents) batch of n rewards for n agents
		# rewards are ordered by agent sequence
		b_rewards = torch.from_numpy(
			np.vstack([e.rewards for e in experiences if e is not None])
			).float().to(device)

		b_dones = torch.from_numpy(
			np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)
			).float().to(device)

		return (b_a_states, b_a_actions, b_a_next_states, b_rewards, b_dones)

	def __len__(self):
		return len(self.memory)







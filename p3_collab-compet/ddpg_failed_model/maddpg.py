
from model import Critic, Actor
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from replay_memory import ReplayMemory

VAR = 1. # initial variance for action random noise
BUFFER_SIZE = 100000 # size for replay memory
BATCH_SIZE = 1024 # size for sampling on replay memory
LR_ACTOR = 1e-3
LR_CRITIC = 1e-2
TAU = 1e-2   # for soft update of target parameters
LEARN_EVERY_STEP = 100
LEARN_REPEAT = 10 # update times per learning
GAMMA = 0.95 # discount factor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG:

	def __init__(self, n_agents, state_size, action_size, seed=299):
		self.seed = random.seed(seed)
		self.n_agents = n_agents
		self.action_size = action_size
		self.batch_size = BATCH_SIZE
		self.t_step = 0  # counter for activating learning every few steps

		self.actors_local = [Actor(state_size, action_size, seed).to(device) for _ in range(n_agents)]
		self.actors_optimizer = [optim.Adam(x.parameters(), lr=LR_ACTOR) for x in self.actors_local]

		self.critics_local = [Critic(state_size, action_size, n_agents, seed).to(device) for _ in range(n_agents)]
		self.critics_optimizer = [optim.Adam(x.parameters(), lr=LR_CRITIC) for x in self.critics_local]

		self.actors_target = [Actor(state_size, action_size, seed).to(device) for _ in range(n_agents)]
		self.critics_target = [Critic(state_size, action_size, n_agents, seed).to(device) for _ in range(n_agents)]

		self.var = [VAR for _ in range(n_agents)] # variance for action exploration
		self.memory = ReplayMemory(BUFFER_SIZE, BATCH_SIZE)

	def act(self, all_states, mode='train'):
		"""
		:param
			all_states (n_agents, state_size) (numpy): states of all agents
			mode (string): 'test' or 'train' mode
		:return:
			actions (n_agents, action_size) (numpy): actions of all agents
		"""
		actions = np.zeros((self.n_agents, self.action_size))

		for i in range(self.n_agents):
			state = torch.from_numpy(all_states[i,:]).unsqueeze(0).float().to(device)

			self.actors_local[i].eval()
			with torch.no_grad():
				act = self.actors_local[i](state).squeeze().cpu().data.numpy()
			self.actors_local[i].train()

			if mode == 'test':
				act = np.clip(act, -1, 1)

			if mode == 'train':
				noise = np.random.randn(self.action_size)*self.var[i]
				act = act + noise
				act = np.clip(act, -1, 1)
				if self.var[i] > 0.05:
					self.var[i] *= 0.999998 # decrease the noise variance after each step

			actions[i,:] = act

		return actions

	def step(self, states, actions, rewards, next_states, dones):
		self.memory.add(states, actions, rewards, next_states, dones)

		# activate learning every few steps
		self.t_step = self.t_step + 1
		if self.t_step % LEARN_EVERY_STEP == 0:
			if len(self.memory) > BATCH_SIZE:
				for _ in range(LEARN_REPEAT):
					experiences = self.memory.sample()
					self.learn(experiences, GAMMA)

	def learn(self, experiences, gamma):

		b_a_states, b_a_actions, b_a_next_states, b_rewards, b_dones = experiences

		all_states = b_a_states.view(self.batch_size, -1) # (batch_size, all_obs)
		all_next_states = b_a_next_states.view(self.batch_size, -1) # (batch_size, all_next_obs)
		all_actions = b_a_actions.view(self.batch_size, -1) # (batch_size, all_act)

		# Get predicted next-state actions and Q values from target models
		for i in range(self.n_agents):
			# ---------------------------- update critic ---------------------------- #
			b_a_next_actions = [
				self.actors_target[k](b_a_next_states[:,k,:].squeeze(1)) for k in range(self.n_agents)
				] # (n_agents, batch_size, state_size)

			b_a_next_actions = torch.stack(b_a_next_actions).float().to(device)

			b_a_next_actions = b_a_next_actions.permute(1,0,2) # (batch_size, n_agents, state_size)
			all_next_actions = b_a_next_actions.contiguous().view(self.batch_size, -1)
			Q_targets_next = self.critics_target[i](all_next_states, all_next_actions) # (batch_size, 1)

			# Compute Q targets for current states (y_i)
			Q_targets = b_rewards[:,i] + (gamma * Q_targets_next * (1 - b_dones[:,i])) # (batch_size, 1)

			# Compute critic loss
			Q_expected = self.critics_local[i](all_states, all_actions) # (batch_size, 1)
			critic_loss = F.mse_loss(Q_expected, Q_targets)

			# Minimize the loss
			self.critics_optimizer[i].zero_grad()
			critic_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.critics_local[i].parameters(), 1)
			self.critics_optimizer[i].step()

			# ------------------- update actor ------------------- #
			# Compute actor loss
			actions_pred = self.actors_local[i](b_a_states[:,i,:].squeeze(1)) # ( batch_size, action_size)
			new_b_a_actions = b_a_actions.clone() # 'clone' create tensor on the same device
			new_b_a_actions[:,i,:] = actions_pred
			new_all_actions = new_b_a_actions.view(self.batch_size, -1)
			actor_loss = -self.critics_local[i](all_states, new_all_actions).mean() # (batch_size, 1)

			# Minimize the loss
			self.actors_optimizer[i].zero_grad()
			actor_loss.backward()
			self.actors_optimizer[i].step()

			# ------------------- update target network ------------------- #
			self.soft_update(self.critics_local[i], self.critics_target[i], TAU)
			self.soft_update(self.actors_local[i], self.actors_target[i], TAU)

	def soft_update(self, local_model, target_model, tau):
		"""Soft update model parameters.
		θ_target = τ*θ_local + (1 - τ)*θ_target
		Params
		======
			local_model: PyTorch model (weights will be copied from)
			target_model: PyTorch model (weights will be copied to)
			tau (float): interpolation parameter
		"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

































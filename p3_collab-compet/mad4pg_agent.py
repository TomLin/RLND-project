'''
The code is partially referred from 1. https://github.com/kelvin84hk/DRLND_P3_collab-compet

'''


from d4pg_memory import ReplayBuffer
from d4pg_agent import D4PG
import numpy as np
import torch
import torch.nn.functional as F

LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
REWARD_STEPS = 1 # steps for rewards of consecutive state action pairs
BUFFER_SIZE = 100000
BATCH_SIZE = 64
N_ATOMS = 51
Vmax = 1
Vmin = -1
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)
TAU = 1e-3 # for soft update of target parameters
LEARN_EVERY_STEP = 10
LEARN_REPEAT = 1


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MAD4PG():

	def __init__(self, state_size, action_size, seed):
		self.t_step = 0
		self.gamma = GAMMA
		self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

		self.mad4pg_agent = [D4PG(state_size, action_size, seed, LR_ACTOR, LR_CRITIC, N_ATOMS, Vmax, Vmin),
							 D4PG(state_size, action_size, seed, LR_ACTOR, LR_CRITIC, N_ATOMS, Vmax, Vmin)]

	def acts(self, states, mode):
		"""
		:param states (n_agents, state_size) (numpy): states for n agents
		:param mode (string): 'test' or 'train'
		:return: acts (n_agents, action_size) (numpy)
		"""
		acts = []
		for s, a in zip(states, self.mad4pg_agent):
			if len(s.shape) < 2:
				s = np.expand_dims(s, axis=0)
			acts.append(a.act(s, mode))
		return np.vstack(acts)

	def learn(self, agent, states, actions, rewards, next_states, dones, gamma):
		"""
		:param agent: one D4PG network
		:param
			states (batch_size, state_size) (tensor)
			actions (batch_size, action_size) (tensor)
			rewards (batch_size,) (tensor)
			next_states (batch_size, state_size) (tensor)
			dones (batch_size,) (tensor)
		:param gamma (float): discount factor
		"""
		# ---------------------------- update critic ---------------------------- #
		Q_expected = agent.critic_local(states, actions)
		actions_next = agent.actor_target(next_states)
		Q_targets_next = agent.critic_target(next_states, actions_next)

		Q_targets_next = F.softmax(Q_targets_next, dim=1)

		proj_distr_v = self.distr_projection(Q_targets_next, rewards, dones,
										gamma=gamma ** REWARD_STEPS, device=device)
		prob_dist_v = -F.log_softmax(Q_expected, dim=1) * proj_distr_v
		critic_loss_v = prob_dist_v.sum(dim=1).mean()

		# Minimize the loss
		agent.critic_optimizer.zero_grad()
		critic_loss_v.backward()
		torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
		agent.critic_optimizer.step()

		# ---------------------------- update actor ---------------------------- #
		# Compute actor loss
		actions_pred = agent.actor_local(states)
		crt_distr_v = agent.critic_local(states, actions_pred)
		actor_loss_v = -agent.critic_local.distr_to_q(crt_distr_v)
		actor_loss_v = actor_loss_v.mean()

		# Minimize the loss
		agent.actor_optimizer.zero_grad()
		actor_loss_v.backward()
		agent.actor_optimizer.step()

		# ------------------- update target network ------------------- #
		agent.soft_update(agent.critic_local, agent.critic_target, TAU)
		agent.soft_update(agent.actor_local, agent.actor_target, TAU)


	def distr_projection(self, next_distr_v, rewards_v, dones_mask_t, gamma, device):
		next_distr = next_distr_v.data.cpu().numpy()
		rewards = rewards_v.data.cpu().numpy()
		dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
		batch_size = len(rewards)
		proj_distr = np.zeros((batch_size, N_ATOMS), dtype=np.float32)

		for atom in range(N_ATOMS):
			tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * DELTA_Z) * gamma))
			b_j = (tz_j - Vmin) / DELTA_Z
			l = np.floor(b_j).astype(np.int64)
			u = np.ceil(b_j).astype(np.int64)
			eq_mask = u == l
			proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
			ne_mask = u != l
			proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
			proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

		if dones_mask.any():
			proj_distr[dones_mask] = 0.0
			tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones_mask]))
			b_j = (tz_j - Vmin) / DELTA_Z
			l = np.floor(b_j).astype(np.int64)
			u = np.ceil(b_j).astype(np.int64)
			eq_mask = u == l
			eq_dones = dones_mask.copy()
			eq_dones[dones_mask] = eq_mask
			if eq_dones.any():
				proj_distr[eq_dones, l[eq_mask]] = 1.0
			ne_mask = u != l
			ne_dones = dones_mask.copy()
			ne_dones[dones_mask] = ne_mask
			if ne_dones.any():
				proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
				proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
		return torch.FloatTensor(proj_distr).to(device)

	def step(self, states, actions, rewards, next_states, dones):
		"""

		:param states (n_agents, state_size) (numpy): agents' state of current timestamp
		:param actions (n_agents, action_size) (numpy): agents' action of current timestamp
		:param rewards (n_agents,):
		:param next_states (n_agents, state_size) (numpy):
		:param dones (n_agents,) (numpy):
		:return:
		"""

		self.memory.add(states, actions, rewards, next_states, dones)

		# activate learning every few steps
		self.t_step = self.t_step + 1
		if self.t_step % LEARN_EVERY_STEP == 0:
			# Learn, if enough samples are available in memory
			if len(self.memory) > BATCH_SIZE:
				for _ in range(LEARN_REPEAT):
					b_a_states, b_a_actions, b_rewards, b_a_next_states, b_dones = self.memory.sample()

					# b_a_states (batch_size, n_agents, states)
					# b_a_actions (batch_size, n_agents, actions)
					# b_rewards (batch_size, n_agents)
					# b_a_next_states (batch_size, n_agents, next_states)
					# b_dones (batch_size, n_agents)

					for i, agent in enumerate(self.mad4pg_agent):
						states = b_a_states[:,i,:].squeeze(1) # (batch_size, state_size)
						actions = b_a_actions[:,i,:].squeeze(1) # (batch_size, action_size)
						rewards = b_rewards[:,i] # (batch_size,)
						next_states = b_a_next_states[:,i,:].squeeze(1) # (batch_size, next_states)
						dones = b_dones[:,i] # (batch_size,)

						self.learn(agent, states, actions, rewards, next_states, dones, GAMMA)
















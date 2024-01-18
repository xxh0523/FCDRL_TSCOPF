import copy
import numpy as np
import torch
import torch.nn as nn
from utils import ReplayBuffer


################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
cur_device = torch.device('cpu')
if(torch.cuda.is_available()): 
    cur_device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(cur_device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# this code is developed based on https://github.com/sfujim/TD3/blob/master/OurDDPG.py 


activation = nn.ReLU


class Critic(nn.Module):
	def __init__(
		self, 
		state_dim, 
		action_dim, 
		hidden_dim=512
		):
		super(Critic, self).__init__()

		self.critic = nn.Sequential(
						nn.Linear(state_dim+action_dim, hidden_dim), activation(),
						nn.Linear(hidden_dim, hidden_dim), activation(),
						nn.Linear(hidden_dim, hidden_dim), activation(),
						nn.Linear(hidden_dim, 1)
					).to(cur_device)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)
		return self.critic(sa)


class Agent_DDPG(object):
	def __init__(
		self, 
		state_dim: int, 
		action_dim: int, 
		hidden_dim=512,
		discount=0.99, 
		actor_lr=1e-5,
		critic_lr=1e-3,
		tau=0.005,
		total_step=20000
	):
		self.actor = nn.Sequential(nn.Linear(state_dim, hidden_dim), activation(),
								   nn.Linear(hidden_dim, hidden_dim), activation(),
								   nn.Linear(hidden_dim, hidden_dim), activation(),
								   nn.Linear(hidden_dim, action_dim), nn.Tanh()
								   ).to(cur_device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
		self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=max(int(total_step * 3 / 4), 1), gamma=0.1)
		# self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=max(int(total_step / 5), 1), gamma=0.618)

		self.critic = Critic(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
		# self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=max(int(total_step / 5), 1), gamma=0.618)
		# self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=max(int(total_step * 1 / 2), 1), gamma=0.1)

		self.discount = discount
		self.tau = tau

	def select_action(self, state: np.ndarray, is_test=True):
		state = torch.FloatTensor(state.reshape(1, -1)).to(cur_device)
		return self.actor(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer: ReplayBuffer, batch_size=256, training_epoch=1):
		for _ in range(training_epoch):
			# Sample replay buffer 
			state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
			# state, action, next_state, reward, not_done = replay_buffer.importance_sampling(batch_size)

			# Compute the target Q value
			target_Q = self.critic_target(next_state, self.actor_target(next_state))
			target_Q = reward + (not_done * self.discount * target_Q).detach()

			# Get current Q estimate
			current_Q = self.critic(state, action)

			# Compute critic loss
			critic_loss = nn.functional.mse_loss(current_Q, target_Q)

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Compute actor loss
			actor_loss = -self.critic(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()
			
			self.actor_scheduler.step()
			# self.critic_scheduler.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		current_model = {}
		
		current_model['critic'] = self.critic.state_dict()
		current_model['critic_optimizer'] = self.critic_optimizer.state_dict()
		
		current_model['actor'] = self.actor.state_dict()
		current_model['actor_optimizer'] = self.actor_optimizer.state_dict()
		
		torch.save(current_model, filename)


	def load(self, filename):
		current_model = torch.load(filename, map_location=cur_device)
		
		self.critic.load_state_dict(current_model['critic'])
		self.critic_optimizer.load_state_dict(current_model['critic_optimizer'])
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(current_model['actor'])
		self.actor_optimizer.load_state_dict(current_model['actor_optimizer'])
		self.actor_target = copy.deepcopy(self.actor)
		
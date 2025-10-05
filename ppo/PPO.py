import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils

class PolicyNet(torch.nn.Module):
  def __init__(self, state_dim, hidden_dim, action_dim):
    super(PolicyNet, self).__init__()
    self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
    self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    return F.softmax(self.fc2(x), dim=1)

class ValueNet(torch.nn.Module):
  def __init__(self, state_dim, hidden_dim):
    super(ValueNet, self).__init__()
    self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
    self.fc2 = torch.nn.Linear(hidden_dim, 1)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    return self.fc2(x)

# test use 'CartPole-v1'
class PPO:
  """PPO算法, 采用截断方式"""
  def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
               lmbda, epochs, eps, gamma, device):
    self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
    self.critic = ValueNet(state_dim, hidden_dim).to(device)
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                            lr = actor_lr)
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                              lr = critic_lr)

    self.gamma = gamma
    self.lmbda = lmbda
    self.epochs = epochs # 一条序列的数据用来训练的轮数
    self.eps = eps
    self.device = device

  def take_action(self, state):
    # print(state.size) # 4
    state = torch.tensor([state], dtype=torch.float).to(self.device)
    # print(state.shape) # [1, 4]
    probs = self.actor(state)
    # print(probs.shape) # [1, 2]
    action_dist = torch.distributions.Categorical(probs)
    action = action_dist.sample()
    # print(action.shape) # [1]
    return action.item()

  def update(self, transition_dict):
    states = torch.tensor(transition_dict['states'],
                          dtype=torch.float).to(self.device)
    # # raw size: (xx, 4), tensor size: [xx, 4], xx: episode horizon
    # print(f'raw states size: {np.array(transition_dict['states']).shape}, tensor size: {states.shape}')
    actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
    # # raw size: (xx,), tensor size: [xx, 1]
    # print(f'raw actions size: {np.array(transition_dict['actions']).shape}, tensor size: {actions.shape}')
    rewards = torch.tensor(transition_dict['rewards'],
                           dtype=torch.float).view(-1, 1).to(self.device)
    # # raw size: (xx,), tensor size: [xx, 1]
    # print(f'raw rewards size: {np.array(transition_dict['rewards']).shape}, tensor size: {rewards.shape}')
    next_states = torch.tensor(transition_dict['next_states'],
                               dtype=torch.float).to(self.device)
    # # raw size: (xx, 4), tensor size: [xx, 4]
    # print(f'raw next_states size: {np.array(transition_dict['next_states']).shape}, tensor size: {next_states.shape}')
    dones = torch.tensor(transition_dict['dones'],
                         dtype=torch.float).view(-1, 1).to(self.device)
    # # raw size: (xx,), tensor size: [xx, 1]
    # print(f'raw dones size: {np.array(transition_dict['dones']).shape}, tensor size: {dones.shape}')

    td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
    td_delta = td_target - self.critic(states)
    advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                           td_delta.cpu()).to(self.device)
    old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

    for _ in range(self.epochs):
      log_probs = torch.log(self.actor(states).gather(1, actions))
      ratio = torch.exp(log_probs - old_log_probs)
      surr1 = ratio * advantage
      surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
      actor_loss = torch.mean(-torch.min(surr1, surr2))
      critic_loss = torch.mean(
          F.mse_loss(self.critic(states), td_target.detach()))
      self.actor_optimizer.zero_grad()
      self.critic_optimizer.zero_grad()
      actor_loss.backward()
      critic_loss.backward()
      self.actor_optimizer.step()
      self.critic_optimizer.step()

class PolicyNetContinuous(torch.nn.Module):
  def __init__(self, state_dim, hidden_dim, action_dim):
    super(PolicyNetContinuous, self).__init__()
    self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
    self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
    self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    mu = 2.0 * torch.tanh(self.fc_mu(x))
    std = F.softplus(self.fc_std(x))
    return mu, std

# test use 'Pendulum-v1'
class PPOContinuous:
  '''处理连续动作的PPO算法'''
  def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
               lmbda, epochs, eps, gamma, device):
    self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
    self.critic = ValueNet(state_dim, hidden_dim).to(device)
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
    self.gamma = gamma
    self.lmbda = lmbda
    self.epochs = epochs
    self.eps = eps
    self.device = device

  def take_action(self, state):
    # print(state.size) # 3
    state = torch.tensor([state], dtype=torch.float).to(self.device)
    # print(state.shape) # [1, 3]
    mu, sigma = self.actor(state)
    # print(f'mu shape: {mu.shape}, sigma shape: {sigma.shape}') # [1, 1], [1, 1]
    action_dist = torch.distributions.Normal(mu, sigma)
    action = action_dist.sample()
    # print(action.shape) # [1, 1]
    return [action.item()]

  def update(self, transition_dict):
    states = torch.tensor(transition_dict['states'],
                          dtype=torch.float).to(self.device)
    # # raw size: (200, 3), tensor size: [200, 4], 200: episode horizon
    # print(f'raw states size: {np.array(transition_dict['states']).shape}, tensor size: {states.shape}')
    actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
    # # raw size: (200, 1), tensor size: [200, 1]
    # print(f'raw actions size: {np.array(transition_dict['actions']).shape}, tensor size: {actions.shape}')
    rewards = torch.tensor(transition_dict['rewards'],
                           dtype=torch.float).view(-1, 1).to(self.device)
    # # raw size: (200,), tensor size: [200, 1]
    # print(f'raw rewards size: {np.array(transition_dict['rewards']).shape}, tensor size: {rewards.shape}')
    next_states = torch.tensor(transition_dict['next_states'],
                               dtype=torch.float).to(self.device)
    # # raw size: (200, 3), tensor size: [200, 3]
    # print(f'raw next_states size: {np.array(transition_dict['next_states']).shape}, tensor size: {next_states.shape}')
    dones = torch.tensor(transition_dict['dones'],
                         dtype=torch.float).view(-1, 1).to(self.device)
    # # raw size: (200,), tensor size: [200, 1]
    # print(f'raw dones size: {np.array(transition_dict['dones']).shape}, tensor size: {dones.shape}')

    rewards = (rewards + 8.0) / 8.0 # 对奖励进行修改，方便训练?
    td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
    td_delta = td_target - self.critic(states)
    advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                           td_delta.cpu()).to(self.device)
    mu, std = self.actor(states)
    action_dists = torch.distributions.Normal(mu.detach(), std.detach())
    # # [200, 1], [200, 1]
    # print(f'mu shape: {mu.shape}, std shape: {std.shape}')
    # # 200维的正太分布
    # print(action_dists)
    # 动作是正太分布
    old_log_probs = action_dists.log_prob(actions)
    # # [200, 1]
    # print(f'old_log_probs shape: {old_log_probs.shape}')

    for _ in range(self.epochs):
      mu, std = self.actor(states)
      action_dists = torch.distributions.Normal(mu, std)
      log_probs = action_dists.log_prob(actions)
      ratio = torch.exp(log_probs - old_log_probs)
      surr1 = ratio * advantage
      surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
      actor_loss = torch.mean(-torch.min(surr1, surr2))
      critic_loss = torch.mean(
          F.mse_loss(self.critic(states), td_target.detach()))
      self.actor_optimizer.zero_grad()
      self.critic_optimizer.zero_grad()
      actor_loss.backward()
      critic_loss.backward()
      self.actor_optimizer.step()
      self.critic_optimizer.step()


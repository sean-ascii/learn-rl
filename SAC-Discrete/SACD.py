import copy
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from utils import DoubleQNet, PolicyNet, ReplayBuffer

class SACDAgent():
  def __init__(self, **kwargs):
    # init hyperparameters for agent, just like "self.gamma=opt.gamme..."
    self.__dict__.update(kwargs)
    self.tau = 0.005
    self.H_mean = 0
    self.replay_buffer = ReplayBuffer(self.state_dim, self.device, max_size=int(1e6))

    self.actor = PolicyNet(self.state_dim, self.action_dim, self.hidden_shape).to(self.device)
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

    self.q_critic = DoubleQNet(self.state_dim, self.action_dim, self.hidden_shape).to(self.device)
    self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.lr)
    self.q_critic_target = copy.deepcopy(self.q_critic)
    for p in self.q_critic_target.parameters(): p.requires_grad = False

    if self.adaptive_alpha:
      # we use 0.6 because the recommended 0.98 will cause alpha explosion
      self.target_entropy = 0.6 * (-np.log(1 / self.action_dim)) # H(discrete) > 0
      self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.device)
      self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)

  def select_action(self, state, deterministic):
    with torch.no_grad():
      state = torch.FloatTensor(state[np.newaxis,:]).to(self.device) # from (s_dim,) to (1, s_dim)
      probs = self.actor(state)
      if deterministic:
        a = probs.argmax(-1).item()
      else:
        a = Categorical(probs).sample().item()
      return a
    
  def train(self):
    s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

    # --------------------------train critic
    '''compute the target soft Q value'''
    with torch.no_grad():
      next_probs = self.actor(s_next) #[b, a_dim]
      next_log_probs = torch.log(next_probs + 1e-8) #[b, a_dim]
      next_q1_all, next_q2_all = self.q_critic_target(s_next) #[b, a_dim]
      min_next_q_all = torch.min(next_q1_all, next_q2_all)
      v_next = torch.sum(next_probs * (min_next_q_all - self.alpha * next_log_probs), dim=1, keepdim=True) #[b, 1]
      target_Q = r + (~dw) * self.gamma * v_next

    '''update soft Q net'''
    q1_all, q2_all = self.q_critic(s) #[b, a_dim]
    q1, q2 = q1_all.gather(1, a), q2_all.gather(1, a) #[b, 1]
    q_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
    self.q_critic_optimizer.zero_grad()
    q_loss.backward()
    self.q_critic_optimizer.step()

    # --------------------------train actor
    probs = self.actor(s) #[b, a_dim]
    log_probs = torch.log(probs + 1e-8) #[b, a_dim]
    with torch.no_grad():
      q1_all, q2_all = self.q_critic(s) #[b, a_dim]
    min_q_all = torch.min(q1_all, q2_all)

    a_loss = torch.sum(probs * (self.alpha * log_probs - min_q_all), dim=1, keepdim=False) #[b,]

    self.actor_optimizer.zero_grad()
    a_loss.mean().backward()
    self.actor_optimizer.step()

    # -------------------------train alpha
    if self.adaptive_alpha:
      with torch.no_grad():
        self.H_mean = -torch.sum(probs * log_probs, dim=1).mean()
      alpha_loss = self.log_alpha * (self.H_mean - self.target_entropy)

      self.alpha_optim.zero_grad()
      alpha_loss.backward()
      self.alpha_optim.step()

      self.alpha = self.log_alpha.exp().item()

    # ---------------------------update target net
    for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
      target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

  def save(self, timestep, env_name):
    torch.save(self.actor.state_dict(), f"./model/sacd_actor_{timestep}_{env_name}.pth")
    torch.save(self.q_critic.state_dict(), f"./model/sacd_critic_{timestep}_{env_name}.pth")

  def load(self, timestep, env_name):
    self.actor.load_state_dict(torch.load(f"./model/sacd_actor_{timestep}_{env_name}.pth", map_location=self.device, weights_only=True))
    self.q_critic.load_state_dict(torch.load(f"./model/sacd_critic_{timestep}_{env_name}.pth", map_location=self.device, weights_only=True))
import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch

def build_net(layer_shape, hidden_activation, output_activation):
  '''build net with for loop'''
  layers = []
  for j in range(len(layer_shape) - 1):
    act = hidden_activation if j < len(layer_shape) - 2 else output_activation
    layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
  return nn.Sequential(*layers)

class DoubleQNet(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_shape):
    super(DoubleQNet, self).__init__()
    layers = [state_dim] + list(hidden_shape) + [action_dim]

    self.Q1 = build_net(layers, nn.ReLU, nn.Identity)
    self.Q2 = build_net(layers, nn.ReLU, nn.Identity)

  def forward(self, s):
    q1 = self.Q1(s)
    q2 = self.Q2(s)
    return q1, q2
  
class PolicyNet(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_shape):
    super(PolicyNet, self).__init__()
    layers = [state_dim] + list(hidden_shape) + [action_dim]
    self.P = build_net(layers, nn.ReLU, nn.Identity)

  def forward(self, s):
    logits = self.P(s)
    probs = F.softmax(logits, dim=1)
    return probs
  
class ReplayBuffer(object):
  def __init__(self, state_dim, device, max_size=int(1e6)):
    self.max_size = max_size
    self.device = device
    self.ptr = 0
    self.size = 0

    self.s = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.device)
    self.a = torch.zeros((max_size, 1), dtype=torch.long, device=self.device)
    self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.device)
    self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.device)
    self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.device)

  def add(self, s, a, r, s_next, dw):
    self.s[self.ptr] = torch.from_numpy(s).to(self.device)
    self.a[self.ptr] = a
    self.r[self.ptr] = r
    self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.device)
    self.dw[self.ptr] = dw

    self.ptr = (self.ptr + 1) % self.max_size
    self.size = min(self.size + 1, self.max_size)

  def sample(self, batch_size):
    index = torch.randint(0, self.size, device=self.device, size=(batch_size,))
    return self.s[index], self.a[index], self.r[index], self.s_next[index], self.dw[index]
  
def evaluate_policy(env, agent, turns = 3):
  total_scores = 0
  for j in range(turns):
    s, info = env.reset()
    done = False
    while not done:
      # take deterministic actions at test time
      a = agent.select_action(s, deterministic=True)
      s_next, r, dw, tr, info = env.step(a)
      done = (dw or tr)

      total_scores += r
      s = s_next
  return int(total_scores/turns)

#You can just ignore 'str2bool'. Is not related to the RL.
def str2bool(v):
	'''transfer str to bool for argparse'''
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
import time
import vpg_utils as utils

import matplotlib.pyplot as plt

class VPGBuffer:
  def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
    self.obs_buf = np.zeros(utils.combined_shape(size, obs_dim), dtype=np.float32)
    self.act_buf = np.zeros(utils.combined_shape(size, act_dim), dtype=np.float32)
    self.adv_buf = np.zeros(size, dtype=np.float32)
    self.rew_buf = np.zeros(size, dtype=np.float32)
    self.ret_buf = np.zeros(size, dtype=np.float32)
    self.val_buf = np.zeros(size, dtype=np.float32)
    self.logp_buf = np.zeros(size, dtype=np.float32)
    self.gamma, self.lam = gamma, lam
    self.ptr, self.path_start_idx, self.max_size = 0, 0, size

  def store(self, obs, act, rew, val, logp):
    assert self.ptr < self.max_size
    self.obs_buf[self.ptr] = obs
    self.act_buf[self.ptr] = act
    self.rew_buf[self.ptr] = rew
    self.val_buf[self.ptr] = val
    self.logp_buf[self.ptr] = logp
    self.ptr += 1

  def finish_path(self, last_val=0):
    """
    Call this at the end of a trajectory, or when one gets cut off by an epoch ending.
    The "last_val" argument should be 0 if the trajectory ended because the agent reached
    a terminal state(died), and otherwise should be V(s_T), the value function estimated for
    the last state.
    """
    path_slice = slice(self.path_start_idx, self.ptr)
    rews = np.append(self.rew_buf[path_slice], last_val)
    vals = np.append(self.val_buf[path_slice], last_val)

    # the next two lines implement GAE-Lambda advantage calculation
    deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
    self.adv_buf[path_slice] = utils.discount_cumsum(deltas, self.gamma * self.lam)

    # the next line computes rewards-to-go, to be targets for the value function
    self.ret_buf[path_slice] = utils.discount_cumsum(rews, self.gamma)[:-1]

    self.path_start_idx = self.ptr

  def get(self):
    """
    Call this at the end of an epoch to get all of the data from
    the buffer, with advantages appropriately normalized (shifted to have
    mean zero and std one). Also, resets some pointers in the buffer.
    """
    assert self.ptr == self.max_size
    self.ptr, self.path_start_idx = 0, 0
    # the next two lines implement the advantage normalization trick
    adv_mean, adv_std = np.average(self.adv_buf), np.std(self.adv_buf)
    self.adv_buf = (self.adv_buf - adv_mean) / adv_std
    data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                adv=self.adv_buf, logp=self.logp_buf)
    return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

GAMMA = 0.99
LAM = 0.97
PI_LR = 3e-4
VF_LR = 1e-3
HIDDEN_SIZE = 64
HIDDEN_LAYER_NUM = 2
STEPS_PER_EPOCH = 4000
EPOCH_NUM = 50
MAX_EPOCH_LEN=1000
TRAIN_V_ITERS = 80

env = gym.make("CartPole-v1")
# env = gym.make("CartPole-v1", render_mode="human")
obs_dim = env.observation_space.shape
act_dim = env.action_space.shape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

actor_critic=utils.MLPActorCritic(env.observation_space, env.action_space,
                                  [HIDDEN_SIZE]*HIDDEN_LAYER_NUM)

var_counts = tuple(utils.count_vars(module) for module in [actor_critic.pi, actor_critic.v])
print("number of parameters: pi:{0}, v:{1}".format(*var_counts))

buf = VPGBuffer(obs_dim, act_dim, STEPS_PER_EPOCH, GAMMA, LAM)

def compute_loss_pi(data):
  obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

  pi, logp = actor_critic.pi(obs, act)
  loss_pi = -(logp * adv).mean()

  return loss_pi

def compute_loss_v(data):
  obs, ret = data['obs'], data['ret']
  return ((actor_critic.v(obs) - ret)**2).mean()

pi_optimizer = Adam(actor_critic.pi.parameters(), lr=PI_LR)
vf_optimizer = Adam(actor_critic.v.parameters(), lr=VF_LR)

loss_pi_list = []
loss_v_list = []
delta_loss_pi_list = []
delta_loss_v_list = []

def update():
  data = buf.get()

  pi_l_old = compute_loss_pi(data).item()
  v_l_old = compute_loss_v(data).item()

  # train policy with a single step of gradient descent
  pi_optimizer.zero_grad()
  loss_pi = compute_loss_pi(data)
  loss_pi.backward()
  pi_optimizer.step()

  # train value function
  for i in range(TRAIN_V_ITERS):
    vf_optimizer.zero_grad()
    loss_v = compute_loss_v(data)
    loss_v.backward()
    vf_optimizer.step()

  loss_pi_list.append(pi_l_old)
  loss_v_list.append(v_l_old)
  delta_loss_pi_list.append(loss_pi.item() - pi_l_old)
  delta_loss_v_list.append(loss_v.item() - v_l_old)

  print("loss pi: {0}, loss v: {1}, delta loss pi: {2}, delta loss v: {3}".format(
      pi_l_old, v_l_old, loss_pi.item() - pi_l_old, loss_v.item() - v_l_old))

  plt.figure(1)
  plt.clf()
  plt.plot(loss_pi_list, label='loss_pi')
  plt.plot(loss_v_list, label='loss_v')
  plt.plot(delta_loss_pi_list, label='delta_loss_pi')
  plt.plot(delta_loss_v_list, label='delta_loss_v')
  plt.legend()
  plt.pause(0.01)

# prepare for interaction with environment
start_time = time.time()
o, _ = env.reset()
epoch_ret = 0
epoch_len = 0

for epoch in range(EPOCH_NUM):
  for t in range(STEPS_PER_EPOCH):
    a, v, logp = actor_critic.step(torch.as_tensor(o, dtype=torch.float32))

    next_o, r, terminated, truncated, _ = env.step(a)
    epoch_ret += r
    epoch_len += 1

    buf.store(o, a, r, v, logp)

    o = next_o

    # done = terminated or truncated # env fail or timeout
    timeout = epoch_len == MAX_EPOCH_LEN # user-specific timeout
    # terminal = done or timeout # the time for reset env
    epoch_ended = t == STEPS_PER_EPOCH-1 # reach max epoch steps

    if terminated or truncated or timeout or epoch_ended:
      if epoch_ended and not(terminated) and not(truncated) and not(timeout):
        print("trajectory cut off by epoch at {0} steps".format(epoch_len))

      # if trajectory didn't reach terminal state, bootstrap value target
      if (timeout or epoch_ended) and not(terminated):
        _, v, _ = actor_critic.step(torch.as_tensor(o, dtype=torch.float32))
      else:
        v = 0
      buf.finish_path(v)
      o, _ = env.reset()
      epoch_ret = 0
      epoch_len = 0

  update()

  print("epoch:{0}".format(epoch))
  print("time:{0}".format(time.time()-start_time))


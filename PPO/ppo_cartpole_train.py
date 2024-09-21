import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
import time
import ppo_utils as utils

class PPOBuffer:
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
    This looks back in the buffer to where the trajectory started, and use rewards
    and value estimates from the whole trajectory to compute advantage estimates with
    GAE-Lambda, as well as compute the rewards-to-go for each state, to use as
    the targets for the value function.

    The "last_val" should be 0 if the trajectory ended because the agent reached a terminal
    state (died), and otherwise should be V(s_T), the value function estimated for the last state.
    This allows us to bootstrap the reward-to-go calculation to account for timesteps
    beyond the arbitrary episode horizon (or epoch cutoff).
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
    Call this at the end of an epoch to get all of the data from the buffer,
    with advantages appropriately normalized (shifted to have mean zero and std one).
    Also, resets some pointers in the buffer.
    """
    assert self.ptr == self.max_size
    self.ptr, self.path_start_idx = 0, 0
    adv_mean, adv_std = np.average(self.adv_buf), np.std(self.adv_buf)
    self.adv_buf = (self.adv_buf - adv_mean) / adv_std
    data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                adv=self.adv_buf, logp=self.logp_buf)
    return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


GAMMA = 0.99
LAM = 0.97
PI_LR = 3e-4
VF_LR= 1e-3
HIDDEN_SIZE = 64
HIDDEN_LAYER_NUM = 2
STEPS_PER_EPOCH = 4000
EPOCH_NUM = 50
MAX_EPOCH_LEN = 1000
TRAIN_PI_ITERS = 80
TRAIN_V_ITERS = 80
TARGET_KL = 0.01
CLIP_RATIO = 0.2


env = gym.make("CartPole-v1")
# env = gym.make("CartPole-v1", render_mode="human")
obs_dim = env.observation_space.shape
act_dim = env.action_space.shape

ac = utils.MLPActorCritic(env.observation_space, env.action_space,
                          [HIDDEN_SIZE]*HIDDEN_LAYER_NUM)

var_counts = tuple(utils.count_vars(module) for module in [ac.pi, ac.v])
print("number of parameters: pi:{0}, v:{1}".format(*var_counts))

buf = PPOBuffer(obs_dim, act_dim, STEPS_PER_EPOCH, GAMMA, LAM)

def compute_loss_pi(data):
  obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

  pi, logp = ac.pi(obs, act)
  ratio = torch.exp(logp - logp_old)
  clip_adv = torch.clamp(ratio, 1-CLIP_RATIO, 1+CLIP_RATIO) * adv
  loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

  # useful extra info
  approx_kl = (logp_old - logp).mean().item()
  ent = pi.entropy().mean().item()
  clipped = ratio.gt(1+CLIP_RATIO) | ratio.lt(1-CLIP_RATIO)
  clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
  pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

  return loss_pi, pi_info

def compute_loss_v(data):
  obs, ret = data['obs'], data['ret']
  return ((ac.v(obs) - ret)**2).mean()


# set up optimizers for policy and value function
pi_optimizer = Adam(ac.pi.parameters(), lr=PI_LR)
vf_optimizer = Adam(ac.v.parameters(), lr=VF_LR)

def update():
  data = buf.get()

  pi_l_old, pi_info_old = compute_loss_pi(data)
  pi_l_old = pi_l_old.item()
  v_l_old = compute_loss_v(data).item()

  for i in range(TRAIN_PI_ITERS):
    pi_optimizer.zero_grad()
    loss_pi, pi_info = compute_loss_pi(data)
    kl = pi_info['kl']
    if kl > 1.5 * TARGET_KL:
      print("early stopping at step: {0} due to reaching max kl.".format(i))
      break
    loss_pi.backward()
    pi_optimizer.step()

  for i in range(TRAIN_V_ITERS):
    vf_optimizer.zero_grad()
    loss_v = compute_loss_v(data)
    loss_v.backward()
    vf_optimizer.step()

  kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
  print("loss pi: {0}, loss v: {1}, kl: {2}, entropy: {3}, clip frac: {4}, delta loss pi: {5}, delta loss v: {6}".format(pi_l_old, v_l_old, kl, ent, cf, (loss_pi.item() - pi_l_old), (loss_v.item() - v_l_old)))

start_time = time.time()
o, _ = env.reset()
ep_ret = 0
ep_len = 0

for epoch in range(EPOCH_NUM):
  for t in range(STEPS_PER_EPOCH):
    a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

    next_o, r, terminated, truncated, _ = env.step(a)
    ep_ret += r
    ep_len += 1

    buf.store(o, a, r, v, logp)

    o = next_o

    # done = terminated or truncated # env fail or timeout
    timeout = ep_len == MAX_EPOCH_LEN # user-specific timeout
    # terminal = done or timeout # the time for reset env
    epoch_ended = t == STEPS_PER_EPOCH-1 # reach max epoch steps

    if terminated or truncated or timeout or epoch_ended:
      if epoch_ended and not(terminated) and not(truncated) and not(timeout):
        print("trajectory cut off by epoch at {0} steps".format(ep_len))
      if (timeout or epoch_ended) and not(terminated):
        _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
      else:
        v = 0
      buf.finish_path(v)
      o, _ = env.reset()
      epoch_ret = 0
      epoch_len = 0

  update()

  print("epoch:{0}".format(epoch))
  print("time:{0}".format(time.time()-start_time))



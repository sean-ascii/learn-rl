from tqdm import tqdm
import numpy as np
import torch
import collections
import random

class ReplayBuffer:
  def __init__(self, capacity):
    self.buffer = collections.deque(maxlen=capacity)

def moving_average(a, window_size):
  pass

def train_on_policy_agent(env, agent, num_episodes):
  return_list = []
  for i in range(10):
    with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
      for i_episode in range(int(num_episodes/10)):
        episode_return = 0
        transition_dict = {'states':[], 'actions':[], 'next_states':[], 'rewards':[], 'dones':[]}
        state, info = env.reset()
        done = False
        while not done:
          action = agent.take_action(state)
          next_state, reward, terminated, truncated, info = env.step(action)
          done = terminated or truncated
          transition_dict['states'].append(state)
          transition_dict['actions'].append(action)
          transition_dict['next_states'].append(next_state)
          transition_dict['rewards'].append(reward)
          # transition_dict['dones'].append(done)
          # 由于无论是terminated还是truncated，都会重置环境并更新策略，因此不用考虑是否基于terminated or truncated重置gae计算
          # 但是，对于next_states的价值评估，terminated后不可用，因此把terminated传递给done，用于td_target计算时不考虑next_state
          # 具体参考“https://www.zhihu.com/question/639133063”
          transition_dict['dones'].append(terminated)
          state = next_state
          episode_return += reward
        return_list.append(episode_return)
        agent.update(transition_dict)
        if (i_episode+1) % 10 == 0:
          pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return' : '%.3f' % np.mean(return_list[-10:])})
        pbar.update(1)

  return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
  pass

def compute_advantage(gamma, lmbda, td_delta):
  td_delta = td_delta.detach().numpy()
  advantage_list = []
  advantage = 0.0
  for delta in td_delta[::-1]:
    advantage = gamma * lmbda * advantage + delta
    advantage_list.append(advantage)
  advantage_list.reverse()
  return torch.tensor(advantage_list, dtype=torch.float)
from tqdm import tqdm
import numpy as np
import torch
import collections
import random

class ReplayBuffer:
  '''经验回放池'''
  def __init__(self, capacity):
    self.buffer = collections.deque(maxlen=capacity)

  def add(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def sample(self, batch_size):
    transitions = random.sample(self.buffer, batch_size)
    # print(transitions)
    state, action, reward, next_state, done = zip(*transitions)
    return np.array(state), action, reward, np.array(next_state), done
  
  def size(self):
    return len(self.buffer)

def moving_average(a, window_size):
  pass

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
  return_list = []
  for i in range(10):
    with tqdm(total=int(num_episodes/10), desc=f'Iteration {i}') as pbar:
      for i_episode in range(int(num_episodes/10)):
        episode_return = 0
        state, info = env.reset()
        done = False
        while not done:
          action = agent.take_action(state)
          next_state, reward, terminated, truncated, info = env.step(action)
          done = terminated or truncated
          replay_buffer.add(state, action, reward, next_state, terminated)
          state = next_state
          episode_return += reward
          if replay_buffer.size() > minimal_size:
            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                               'dones': b_d}
            agent.update(transition_dict)
        return_list.append(episode_return)
        if (i_episode+1) % 10 == 0:
          pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1),
                            'return': '%.3f' % np.mean(return_list[-10:])})
        pbar.update(1)
  return return_list

# 离散动作转回连续的函数
def dis_to_con(discrete_action, env, action_dim):
  action_lowbound = env.action_space.low[0] # 连续动作的最小值
  action_upbound = env.action_space.high[0] # 连续动作的最大值
  return action_lowbound + (discrete_action / (action_dim - 1)) * (action_upbound - action_lowbound)

def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size):
  return_list = []
  max_q_value_list = []
  max_q_value = 0
  for i in range(10):
    with tqdm(total=int(num_episodes/10), desc=f'Iteration {i}') as pbar:
      for i_episode in range(int(num_episodes/10)):
        episode_return = 0
        state, info = env.reset()
        done = False
        while not done:
          action = agent.take_action(state)
          max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995 # 平滑处理
          max_q_value_list.append(max_q_value) # 保存每个状态的最大Q值
          action_continuous = dis_to_con(action, env, agent.action_dim)
          # 对于 'Pendulum-v1', 输入是一个包含单个浮点数的列表
          next_state, reward, terminated, truncated, info = env.step([action_continuous])
          done = terminated or truncated
          replay_buffer.add(state, action, reward, next_state, terminated)
          state = next_state
          episode_return += reward
          if replay_buffer.size() > minimal_size:
            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                               'dones': b_d}
            agent.update(transition_dict)
        return_list.append(episode_return)
        if (i_episode+1) % 10 == 0:
          pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1),
                            'return': '%.3f' % np.mean(return_list[-10:])})
        pbar.update(1)
  return return_list, max_q_value_list

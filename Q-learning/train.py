from q_learning import QLearningAgent, evaluate_policy
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import TimeLimit
from datetime import datetime
import gymnasium as gym
import numpy as np
import os, shutil

def main():
  write = True # whether use SummaryWriter to record training curve
  load_model = False # load model or not
  max_train_steps = 20000
  seed = 0
  np.random.seed(seed)
  print(f"random seed: {seed}")

  """build env"""
  env_name = 'CliffWalking-v0'
  env = gym.make(env_name)
  env = TimeLimit(env, max_episode_steps=500)
  eval_env = gym.make(env_name)
  # eval_env = gym.make(env_name, render_mode="human")
  eval_env = TimeLimit(eval_env, max_episode_steps=100)

  """use tensorboard to record training curves"""
  if write:
    timenow = str(datetime.now())[0:-7]
    timenow = ' ' + timenow[0:13] + '_' + timenow[14:16]+ '_' + timenow[-2::]
    writepath = 'runs/{}'.format(env_name) + timenow
    if os.path.exists(writepath): shutil.rmtree(writepath)
    writer = SummaryWriter(log_dir=writepath)

  """build Q-learning agent"""
  if not os.path.exists('model'):
    os.mkdir('model')
  agent = QLearningAgent(
      s_dim=env.observation_space.n,
      a_dim=env.action_space.n,
      lr=0.2,
      gamma=0.9,
      exp_noise=0.1)
  if load_model: agent.restore()

  """train"""
  total_steps = 0
  while total_steps < max_train_steps:
    s, info = env.reset(seed=seed)
    seed += 1
    done, steps = False, 0

    while not done:
      steps += 1
      a = agent.select_action(s, deterministic=False)
      s_next, r, dw, tr, info = env.step(a)
      agent.train(s, a, r, s_next, dw)

      done = (dw or tr)
      s = s_next

      total_steps += 1
      """record & log"""
      if total_steps % 100 == 0:
        ep_r = evaluate_policy(eval_env, agent)
        if write:
          writer.add_scalar('ep_r', ep_r, global_step=total_steps)
          print(f'EnvName:{env_name}, Seed:{seed}, Steps:{total_steps}, Episode reward:{ep_r}')

      """save model"""
      if total_steps % max_train_steps == 0:
        agent.save()

  env.close()
  eval_env.close()

if __name__ == '__main__':
  main()
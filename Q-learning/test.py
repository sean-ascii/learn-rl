from q_learning import QLearningAgent, evaluate_policy
import gymnasium as gym
import numpy as np

def main():
  env_name = 'CliffWalking-v0'
  eval_env = gym.make(env_name, render_mode="human")

  agent = QLearningAgent(
    s_dim=eval_env.observation_space.n,
    a_dim=eval_env.action_space.n,
    lr=0.2,
    gamma=0.9,
    exp_noise=0.1)
  agent.restore()

  ep_r = evaluate_policy(eval_env, agent)
  print(f'evaluate return: {ep_r}')

if __name__ == '__main__':
  main()
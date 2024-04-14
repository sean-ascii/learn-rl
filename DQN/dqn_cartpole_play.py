import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
  def __init__(self, n_observations, n_actions):
    super(DQN, self).__init__()
    self.layer1 = nn.Linear(n_observations, 128)
    self.layer2 = nn.Linear(128, 128)
    self.layer3 = nn.Linear(128, n_actions)

  def forward(self, x):
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    return self.layer3(x)

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()


# policy_net = torch.load('./model.pth', map_location='cpu')
policy_net = DQN(len(observation), env.action_space.n)
state_dict = torch.load('./model_state_dict.pth', map_location='cpu')
policy_net.load_state_dict(state_dict)

for _ in range(1000):
    state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    action = policy_net(state).max(1).indices.view(1, 1).item()
    observation, reward, terminated, truncated, info = env.step(action)

    # print(observation)
    # print(reward)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
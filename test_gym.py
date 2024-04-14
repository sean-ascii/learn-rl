import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    print(observation)
    print(reward)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
from utils import evaluate_policy, str2bool
from datetime import datetime
from PPO import PPO_discrete
import gymnasium as gym
import os, shutil
import argparse
import torch

'''hyperparameter setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', help='running device: cuda or cpu')
parser.add_argument('--env_index', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--load_model', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--model_index', type=int, default=300000, help='which model to load')

parser.add_argument('--seed', type=int, default=209, help='random seed')
parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
parser.add_argument('--max_train_steps', type=int, default=1e6, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=5e3, help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--k_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--l2_reg', type=float, default=0, help='L2 regulization coefficient for Critic')
parser.add_argument('--batch_size', type=int, default=64, help='lenth of sliced trajectory')
parser.add_argument('--entropy_coef', type=float, default=0, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
parser.add_argument('--adv_normalization', type=str2bool, default=False, help='Advantage normalization')
opt = parser.parse_args()
opt.device = torch.device(opt.device) # from str to torch.device
print(opt)

def main():
  # build training env and evaluation env
  EnvName = ['CartPole-v1', 'LunarLander-v2']
  BriefEnvName = ['CP-v1', 'LLd-v2']
  env = gym.make(EnvName[opt.env_index], render_mode = "human" if opt.render else None)
  eval_env = gym.make(EnvName[opt.env_index])
  opt.state_dim = env.observation_space.shape[0]
  opt.action_dim = env.action_space.n
  opt.max_e_steps = env._max_episode_steps

  # seed everything
  env_seed = opt.seed
  torch.manual_seed(opt.seed)
  torch.cuda.manual_seed(opt.seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  print('random seed: {}'.format(opt.seed))

  print('env: ', BriefEnvName[opt.env_index], ', state_dim: ', opt.state_dim, ', action_dim: ', opt.action_dim,
        ', rand seed: ', opt.seed, ', max_e_steps: ', opt.max_e_steps)
  print('\n')

  # use tensorboard to record traing curves
  if opt.write:
    from torch.utils.tensorboard import SummaryWriter
    timenow = str(datetime.now())[0:-10]
    timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
    writepath = 'runs/{}'.format(BriefEnvName[opt.env_index]) + timenow
    if os.path.exists(writepath): shutil.rmtree(writepath)
    writer = SummaryWriter(log_dir=writepath)

  if not os.path.exists('model'): os.mkdir('model')
  agent = PPO_discrete(**vars(opt))
  if opt.load_model: agent.load(opt.model_index)

  if opt.render:
    while True:
      ep_r = evaluate_policy(env, agent, turns=1)
      print(f'Env:{EnvName[opt.env_index]}, Episode Reward:{ep_r}')
  else:
    traj_length, total_steps = 0, 0
    while total_steps < opt.max_train_steps:
      s, info = env.reset(seed=env_seed)
      env_seed += 1
      done = False

      '''interact and train'''
      while not done:
        '''interact with env'''
        a, logprob_a = agent.select_action(s, deterministic=False)
        s_next, r, dw, tr, info = env.step(a) # dw: dead&win, tr: truncated
        if r <= -100: r = -30 # good for LunarLander
        done = (dw or tr)

        '''store the current transition'''
        agent.put_data(s, a, r, s_next, logprob_a, done, dw, idx=traj_length)
        s = s_next

        traj_length += 1
        total_steps += 1

        '''update if its time'''
        if traj_length % opt.T_horizon == 0:
          agent.train()
          traj_length = 0

        '''record & log'''
        if total_steps % opt.eval_interval == 0:
          score = evaluate_policy(eval_env, agent, turns=3) # evaluate the policy for 3 times, and get averaged result
          if opt.write: writer.add_scalar('ep_r', score, global_step=total_steps)
          print('EnvName: ', EnvName[opt.env_index], ', seed: ', opt.seed, ', steps: {}k'.format(int(total_steps/1000)), ', score: ', score)

        '''save model'''
        if total_steps % opt.save_interval == 0:
          agent.save(total_steps)

  env.close()
  eval_env.close()

if __name__ == '__main__':
  main()

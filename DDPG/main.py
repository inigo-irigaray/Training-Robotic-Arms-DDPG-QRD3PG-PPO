import argparse
import os
import time
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import numpy as np

from collections import deque
from pathlib import Path
from unityagents import UnityEnvironment

import buffer
import ddpg




def run(config):
    model_dir = Path('./DDPG/')
    if not model_dir.exists():
        current_run = 'run1'
    else:
        run_nums = [int(str(folder.name).split('run')[1]) 
                        for folder in model_dir.iterdir() if str(folder.name).startswith('run')]
        if len(run_nums) == 0:
            current_run = 'run1'
        else:
            current_run = 'run%i' % (max(run_nums) + 1)
            
    run_dir = model_dir / current_run
    logs_dir = run_dir / 'logs'
    os.makedirs(logs_dir)
    
    writer = SummaryWriter(str(logs_dir))
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    if torch.cuda.is_available() and config.cuda==True:
        cuda = True
    else:
        cuda = False
    env = UnityEnvironment(file_name=config.env)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    
    ddpg = ddpg.DDPGAgent.init_from_env(env_info, brain, hid1=config.hid1, hid2=config.hid2, norm=config.norm,
                                        lr=config.lr, epsilon=config.epsilon, gamma=config.gamma, tau=config.tau)
    print(ddpg.actor)
    print(ddpg.critic)
    repbuffer = buffer.ReplayBuffer(capacity=config.capacity, num_agents=num_agents)
    episode = 0
    reward_100 = deque(maxlen=100)
    while True:
        env_info = env.reset(train_mode=True)[brain_name]
        obs = env_info.vector_observations
        ddpg.prep_rollouts(device='cpu')
        total_rewards = np.zeros(num_agents)
        t = time.time()
        it = 0
        while True:
            it += 1
            obs = torch.FloatTensor(np.vstack(obs))
            actions = ddpg.step(obs, explore=True)
            env_info = env.step(actions)[brain_name]
            next_obs = env_info.vector_observations
            rewards = env_info.rewards
            total_rewards += rewards
            dones = env_info.local_done
            
            repbuffer.add(obs, actions, rewards, next_obs, dones)
            
            if np.any(dones):
                mean_reward = np.mean(total_rewards)
                writer.add_scalar('mean_episode_reward', mean_reward, episode)
                print("Done episode %d for an average reward of %.3f in %.2f seconds, iteration %d." % 
                      (episode, mean_reward, (time.time() - t), it))
                t = time.time()
                reward_100.append(mean_reward)
                break
                        
            obs = next_obs
            if repbuffer.filled > config.batch_size:
                if cuda:
                    ddpg.prep_training(device='gpu')
                else:
                    ddpg.prep_training(device='cpu')
                    
                sample = repbuffer.sample(batch_size=config.batch_size, to_gpu=True)
                ddpg.update(sample, writer=writer)
                ddpg.prep_rollouts(device='cpu')
            
        episode += 1
        if np.mean(reward_100) >= 30.0:
            print("Solved the environment in %d episodes and %.2f minutes." % (episode, (time.time() / 60)))
            env.close()
            break
            
            
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='/data/Reacher_Linux_NoVis/Reacher.x86_64',
                        required=True, help='Path to environment file.', type=str)
    parser.add_argument('--cuda', default=False, type=bool)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--hid1', default=400, type=int)
    parser.add_argument('--hid2', default=300, type=int)
    parser.add_argument('--norm', default='layer', type=str, help="Normalization layer takes values 'batch' for BatchNorm, \
                                                                   'layer' for LayerNorm, and any other value for no \
                                                                    normalization layer.")
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epsilon', default=0.3, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--tau', default=1e-3, type=float)
    parser.add_argument('--capacity', default=100000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)

    config = parser.parse_args()
    
    run(config)

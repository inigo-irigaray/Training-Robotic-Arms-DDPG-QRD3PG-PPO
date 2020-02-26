import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from collections import OrderedDict




class Actor(nn.Module):
    def __init__(self, obs_size, act_size, hid1=400, hid2=300, norm='layer'):
        super(Actor, self).__init__()
        self.net = nn.Sequential()
        if norm=='batch':
            self.net.add_module('bn1', nn.BatchNorm1d(obs_size))
        elif norm=='layer':
            self.net.add_module('ln1', nn.LayerNorm(obs_size))
        else:
            pass
        self.net.add_module('fc1', nn.Linear(obs_size, hid1))
        self.net.add_module('nl1', nn.ReLU())
        self.net.add_module('fc2', nn.Linear(hid1, hid2))
        self.net.add_module('nl2', nn.ReLU())
        self.net.add_module('fc3', nn.Linear(hid2, act_size))
        self.net.add_module('nl3', nn.Tanh())
        
    def forward(self, x):
        return self.net(x)
    
    
    
    
class Critic(nn.Module):
    def __init__(self, obs_size, act_size, hid1=400, hid2=300, norm='layer'):
        super(Critic, self).__init__()
        self.net1 = nn.Sequential()
        if norm=='batch':
            self.net1.add_module('bn1', nn.BatchNorm1d(obs_size))
        elif norm=='layer':
            self.net1.add_module('ln1', nn.LayerNorm(obs_size))
        else:
            pass
        self.net1.add_module('fc1', nn.Linear(obs_size, hid1))
        self.net1.add_module('nl1', nn.ReLU())
        
        self.net2 = nn.Sequential()
        self.net2.add_module('fc2', nn.Linear(hid1 + act_size, hid2))
        self.net2.add_module('nl2', nn.ReLU())
        self.net2.add_module('fc3', nn.Linear(hid2, 1))
        
    def forward(self, x, a):
        o = self.net1(x)
        return self.net2(torch.cat([o, a], dim=1))
    
    
    
    
class TargetModel:
    def __init__(self, model):
        self.model = model
        self.tgt_model = copy.deepcopy(self.model)
    
    def hard_update(self):
        self.tgt_model.load_state_dict(self.model.state_dict())
        
    def soft_update(self, tau):
        assert isinstance(tau, float)
        assert 0. < tau <= 1.
        state = self.model.state_dict()
        tgt_state = self.tgt_model.state_dict()
        for key, value in state.items():
            tgt_state[key] = tgt_state[key] * (1 - tau) + value * tau
        self.tgt_model.load_state_dict(tgt_state)
        
        
        
        
class DDPGAgent:
    def __init__(self, obs_size, act_size, hid1=400, hid2=300, norm='layer', lr=0.01, epsilon=0.3, gamma=0.99, tau=0.01):
        self.actor = Actor(obs_size, act_size, hid1=hid1, hid2=hid2, norm=norm)
        self.tgt_actor = TargetModel(self.actor)
        self.critic = Critic(obs_size, act_size, hid1=hid1, hid2=hid2, norm=norm)
        self.tgt_critic = TargetModel(self.critic)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = tau
        self.iter = 0
        self.time = time.time()
        self.actor_dev = 'cpu'
        self.tgt_actor_dev = 'cpu'
        self.critic_dev = 'cpu'
        self.tgt_critic_dev = 'cpu'
        
    def step(self, obs, explore=True):
        action = self.actor(obs)
        action = action.data.cpu().numpy()
        if explore:
            action += self.epsilon * np.random.normal(size=action.shape)
        action = np.clip(action, -1, 1)
        return action
        
    def update(self, sample, buffer, writer=None):
        self.iter += 1
        obs, action, reward, next_obs, done, idxs, weights = sample
        
        # train critic 
        self.critic_optim.zero_grad()
        next_action = self.tgt_actor.tgt_model(next_obs)
        qnext = self.tgt_critic.tgt_model(next_obs, next_action)
        qnext[done] = 0.0
        next_val = reward + self.gamma * qnext
        qval = self.critic(obs, action)
        critic_loss = F.mse_loss(qval, next_val.detach())
        critic_loss.backward()
        self.critic_optim.step()
        
        # train actor
        self.actor_optim.zero_grad()
        actions = self.actor(obs)
        actor_loss = -self.critic(obs, actions)
        sample_prios = np.abs(actor_loss.data.cpu().numpy()) + 1e-5
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.actor_optim.step()
        
        if writer:
            writer.add_scalar('critic_loss', critic_loss, self.iter)
            writer.add_scalar('actor_loss', actor_loss, self.iter)
            speed = 1 / (time.time() - self.time)
            self.time = time.time()
            writer.add_scalar('speed-iters/sec', speed, self.iter)
            writer.add_scalar('iterations', self.iter, self.time)
           
        buffer.update_priorities(idxs, sample_prios)
            
        # update target models
        self.tgt_actor.soft_update(self.tau)
        self.tgt_critic.soft_update(self.tau)
                
    def prep_training(self, device='gpu'):
        self.actor.train()
        self.tgt_actor.tgt_model.train()
        self.critic.train()
        self.tgt_critic.tgt_model.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.actor_dev == device:
            self.actor = fn(self.actor)
            self.actor_dev = device
        if not self.tgt_actor_dev == device:
            self.tgt_actor.tgt_model = fn(self.tgt_actor.tgt_model)
            self.tgt_actor_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.tgt_critic_dev == device:
            self.tgt_critic.tgt_model = fn(self.tgt_critic.tgt_model)
            self.actor_dev = device
            
    def prep_rollouts(self, device='cpu'):
        self.actor.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.actor_dev == device:
            self.actor = fn(self.actor)
            self.actor_dev = device
     
    def save(self, filename):
        self.prep_training(device='cpu') # move parameters to CPU before saving
        save_dict = {'actor': self.actor.state_dict(),
                     'critic': self.critic.state_dict(),
                     'tgt_actor': self.tgt_actor.tgt_model.state_dict(),
                     'tgt_critic': self.tgt_critic.tgt_model.state_dict(),
                     'actor_optim': self.actor_optim.state_dict(),
                     'critic_optim': self.critic_optim.state_dict(),
                     'init_dict': self.init_dict}
        torch.save(save_dict, filename)
     
    @classmethod
    def init_from_env(cls, env_info, brain, hid1=400, hid2=300, norm='layer', lr=1e-4, epsilon=0.3, gamma=0.99, tau=1e-3):
        obspace = env_info.vector_observations.shape[1]
        aspace = brain.vector_action_space_size
        obs_size = obspace
        act_size = aspace
        init_dict = {'obs_size': obs_size,
                     'act_size': act_size,
                     'hid1': hid1,
                     'hid2': hid2,
                     'norm': norm,
                     'lr': lr,
                     'epsilon': epsilon,
                     'gamma': gamma,
                     'tau': tau,
                     }
        
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        
        return instance
        
    @classmethod
    def init_from_save(cls, filename):
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        instance.actor.load_state_dict(save_dict['actor'])
        instance.tgt_actor.tgt_model.load_state_dict(save_dict['tgt_actor'])
        instance.critic.load_state_dict(save_dict['critic'])
        instance.tgt_critic.tgt_model.load_state_dict(save_dict['tgt_critic'])
        instance.actor_optim.load_state_dict(save_dict['actor_optim'])
        instance.critic_optim.load_state_dict(save_dict['critic_optim'])

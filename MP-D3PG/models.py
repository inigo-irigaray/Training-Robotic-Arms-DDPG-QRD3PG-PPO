import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




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

    @classmethod
    def init_from_env(cls, env_info, brain, hid1=400, hid2=300, norm='layer', lr=1e-4, epsilon=0.3,
                      gamma=0.99, tau=1e-3):
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

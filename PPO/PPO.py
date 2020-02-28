import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F




class Trajectory:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.dones = []
        
    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.logprobs.clear()
        self.dones.clear()
    
    
    

class Actor(nn.Module):
    def __init__(self, obs_size, act_size, num_agents, hid1=400, hid2=300, norm='layer'):
        super(Actor, self).__init__()
        self.mu = nn.Sequential()
        if norm=='batch':
            self.mu.add_module('bn1', nn.BatchNorm1d(obs_size))
        elif norm=='layer':
            self.mu.add_module('ln1', nn.LayerNorm(obs_size))
        else:
            pass
        self.mu.add_module('fc1', nn.Linear(obs_size, hid1))
        self.mu.add_module('nl1', nn.Tanh())
        self.mu.add_module('fc2', nn.Linear(hid1, hid2))
        self.mu.add_module('nl2', nn.Tanh())
        self.mu.add_module('fc3', nn.Linear(hid2, act_size))
        self.mu.add_module('nl3', nn.Tanh())
        
        self.logstd = nn.Parameter(torch.zeros(act_size))
        
    def forward(self, x):
        return self.mu(x)

    
    
    
class Critic(nn.Module):
    def __init__(self, obs_size, hid1=400, hid2=300, norm='layer'):
        super(Critic, self).__init__()
        self.value = nn.Sequential()
        if norm=='batch':
            self.value.add_module('bn1', nn.BatchNorm1d(obs_size))
        elif norm=='layer':
            self.value.add_module('ln1', nn.LayerNorm(obs_size))
        else:
            pass
        self.value.add_module('fc1', nn.Linear(obs_size, hid1))
        self.value.add_module('nl1', nn.ReLU())
        self.value.add_module('fc2', nn.Linear(hid1, hid2))
        self.value.add_module('nl2', nn.ReLU())
        self.value.add_module('fc3', nn.Linear(hid2, 1))
        
    def forward(self, x):
        return self.value(x).squeeze(-1)
        
    
    
    
class PPOAgent:
    def __init__(self, obs_size, act_size, num_agents, hid1=400, hid2=300, norm='layer', lr=1e-4,
                 gamma=0.99, gae_lambda=0.95, epochs=1, eps=0.01):
        self.actor = Actor(obs_size, act_size, num_agents, hid1=hid1, hid2=hid2, norm=norm)
        self.old_actor = copy.deepcopy(self.actor)
        self.critic = Critic(obs_size, hid1=hid1, hid2=hid2, norm=norm)
        self.optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        self.eps = eps
        self.actor_dev = 'cpu'
        self.old_actor_dev = 'cpu'
        self.critic_dev = 'cpu'
        
    def step(self, obs, trajectory, explore=True):
        mu = self.old_actor(obs)
        dist = torch.distributions.normal.Normal(mu, F.softplus(self.old_actor.logstd))
        actions = dist.sample()
        logprobs = dist.log_prob(actions)
        
        trajectory.obs.append(obs)
        trajectory.actions.append(actions)
        trajectory.logprobs.append(logprobs)
        
        return actions.detach()
    
    def evaluate(self, obs, actions):
        mu = self.actor(obs)
        dist = torch.distributions.normal.Normal(mu, F.softplus(self.actor.logstd))
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(obs)
                
        return logprobs, torch.squeeze(values), entropy
    
    def gae(self, obs, trajectory):
        # converts critic output shape from (update_timesteps, num_agents, 1) to (update_timesteps, num_agents) 
        values = self.critic(obs).data.cpu().numpy()
        
        last_gae = 0.0
        adv_est = []
        adv_ref = []
        est = []
        ref = []
        for val, next_val, done, reward in zip(reversed(values[:-1]), reversed(values[1:]),
                                               reversed(trajectory.dones[:-1]), reversed(trajectory.rewards[:-1])):
            if np.any(done):
                delta = reward - val
                last_gae = delta
            else:
                delta = reward + self.gamma * next_val - val
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
            adv_est.append(last_gae)
            adv_ref.append(last_gae + val)
            
        adv_est = torch.FloatTensor(list(reversed(adv_est))).to(self.actor_dev)
        adv_ref = torch.FloatTensor(list(reversed(adv_ref))).to(self.actor_dev)
        return adv_est, adv_ref
    
    def logprobs(self, mu, logstd, actions):
        p1 = -((mu - actions) ** 2) / (2*torch.exp(logstd).clamp(min=1e-3))
        p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd)))
        return p1 + p2
     
    def update(self, trajectory, writer=None, idx=None):
        # converts lists to tensors
        old_obs = torch.squeeze(torch.stack(trajectory.obs).to(self.actor_dev), 1).detach()
        old_actions = torch.squeeze(torch.stack(trajectory.actions).to(self.actor_dev), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(trajectory.logprobs), 1).to(self.actor_dev).detach()
        
        # calculates and normalizes advantages
        adv_est, adv_ref = self.gae(old_obs, trajectory)
        adv_est = (adv_est - torch.mean(adv_est)) / torch.std(adv_est)
        adv_est = adv_est.unsqueeze(-1)
        
        for _ in range(self.epochs):
            # evaluates old actions and observations
            logprobs, values, entropy = self.evaluate(old_obs, old_actions)
            
            self.optim.zero_grad()
            
            # critic training
            
            critic_loss = 0.5 * F.mse_loss(values[:-1], adv_ref)

            # actor training
            
            ratio = torch.exp(logprobs[:-1] - old_logprobs[:-1])
            surr_obj = adv_est * ratio
            clipped_surr = adv_est * torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps)
            actor_loss = -torch.min(surr_obj, clipped_surr).mean() - 0.01 * (torch.sum(entropy, dim=-1)).mean()
            loss = actor_loss + critic_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.actor.parameters(), 0.75)
            torch.nn.utils.clip_grad_norm(self.critic.parameters(), 0.75)
            self.optim.step()
            
            if writer is not None:
                writer.add_scalar("actor_loss", actor_loss, idx)
                writer.add_scalar("critic_loss", critic_loss, idx)
            
        # load new weights into old policy
        self.old_actor.load_state_dict(self.actor.state_dict())
        
        writer.add_scalar("advantage", adv_est.mean().item(), idx)
        writer.add_scalar("values", adv_ref.mean().item(), idx)
        
    def prep_training(self, device='cuda'):
        self.actor.train()
        self.old_actor.train()
        self.critic.train()
        if device == 'cuda':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.actor_dev == device:
            self.actor = fn(self.actor)
            self.actor_dev = device
        if not self.old_actor_dev == device:
            self.old_actor = fn(self.old_actor)
            self.old_actor_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
            
    def prep_rollout(self, device='cpu'):
        self.old_actor.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.old_actor_dev == device:
            self.old_actor = fn(self.old_actor)
            self.old_actor_dev = device
        
    def save(self, filename):
        save_dict = {'actor': self.actor.state_dict(),
                     'critic': self.critic.state_dict(),
                     'old_actor': self.old_actor.state_dict(),
                     'optim': self.optim.state_dict(),
                     'init_dict': self.init_dict}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env_info, brain, hid1=400, hid2=300, norm='layer',
                      lr=1e-4, gamma=0.99, gae_lambda=0.95, epochs=1, eps=0.01):
        obs_size = env_info.vector_observations.shape[1]
        act_size = brain.vector_action_space_size
        num_agents = len(env_info.agents)
        init_dict = {'obs_size': obs_size,
                     'act_size': act_size,
                     'num_agents': num_agents,
                     'hid1': hid1,
                     'hid2': hid2,
                     'norm': norm,
                     'lr': lr,
                     'gamma': gamma,
                     'gae_lambda': gae_lambda,
                     'epochs': epochs,
                     'eps': eps,
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
        instance.old_actor.load_state_dict(save_dict['old_actor'])
        instance.critic.load_state_dict(save_dict['critic'])
        instance.optim.load_state_dict(save_dict['optim'])
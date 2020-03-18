import time
import queue

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import Critic, TargetModel




class Learner:
    def __init__(self, obs_size, act_size, hid1, hid2, norm, actor,
                 tgt_actor, lr, learner_queue, gamma, tau):
        self.critic = Critic(obs_size, act_size, hid1=hid1, hid2=hid2, norm=norm)
        self.tgt_critic = TargetModel(self.critic)
        self.actor = actor
        self.tgt_actor = tgt_actor
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.actor_dev = 'cpu'
        self.tgt_actor_dev = 'cpu'
        self.critic_dev = 'cpu'
        self.tgt_critic_dev = 'cpu'

        # self.num_train_steps =
        self.learner_queue = learner_queue
        self.gamma = gamma
        self.tau = tau
        self.time = time.time()


    def _update(self, sample, update_step, writer=None):
        obs, action, reward, next_obs, done = sample

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
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.actor_optim.step()

        if writer:
            iter = update_step.value
            writer.add_scalar('critic_loss', critic_loss, iter)
            writer.add_scalar('actor_loss', actor_loss, iter)
            speed = 1 / (time.time() - self.time)
            self.time = time.time()
            writer.add_scalar('speed-iters/sec', speed, iter)
            writer.add_scalar('iterations', iter, self.time)

        # update target models
        self.tgt_actor.soft_update(self.tau)
        self.tgt_critic.soft_update(self.tau)

        # Send updated learner actor weights to the queue to update agents' local actors
        if update_step.value % 100 == 0:
            try:
                params = [p.data.to(self.actor_dev).detach().numpy()
                          for p in self.actor.parameters()]
                self.learner_queue.put_nowait(params)
            except:
                pass

    def _prep_training(self, device='gpu'):
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

    def _save(self, filename):
        self._prep_training(device='cpu') # move parameters to CPU before saving
        save_dict = {'actor': self.actor.state_dict(),
                     'critic': self.critic.state_dict(),
                     'tgt_actor': self.tgt_actor.tgt_model.state_dict(),
                     'tgt_critic': self.tgt_critic.tgt_model.state_dict(),
                     'actor_optim': self.actor_optim.state_dict(),
                     'critic_optim': self.critic_optim.state_dict()}
        torch.save(save_dict, filename)

    def run(self, batch_queue, update_step, train, device='cpu', writer=None, filename=""): # training_on
        self._prep_training(device=device)
        while True: # while update_step.value < self.num_train_steps: this implies i let it run for x number of steps
            try:
                sample = batch_queue.get_nowait()
            except queue.Empty:
                continue

            self._update(sample, update_step, writer)
            update_step.value += 1

        self._save(filename)
        train.value = False

        # Clears learner_queue
        while True:
            try:
                clearer = self.learner_queue.get_nowait()
                del clearer
            except:
                break
        self.learner_queue.close()

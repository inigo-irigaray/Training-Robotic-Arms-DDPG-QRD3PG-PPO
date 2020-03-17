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

        self.num_train_steps = 1############ ASSIGMENT PENDING!!!!!
        self.learner_queue = learner_queue
        self.gamma = gamma
        self.tau = tau
        self.iter = 0
        self.time = time.time()


    def update(self, sample, update_step, writer=None):
        self.iter += 1
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
            writer.add_scalar('critic_loss', critic_loss, self.iter)
            writer.add_scalar('actor_loss', actor_loss, self.iter)
            speed = 1 / (time.time() - self.time)
            self.time = time.time()
            writer.add_scalar('speed-iters/sec', speed, self.iter)
            writer.add_scalar('iterations', self.iter, self.time)

        # update target models
        self.tgt_actor.soft_update(self.tau)
        self.tgt_critic.soft_update(self.tau)

        # Send updated learner to the queue
        if update_step.value % 100 == 0:
            try:
                params = [p.data.to(self.actor_dev).detach().numpy()
                          for p in self.actor.parameters()]
                self.learner_queue.put_nowait(params)
            except:
                pass

    def run(self, training_on, batch_queue, update_step):
        while update_step.value < self.num_train_steps:
            try:
                batch = batch_queue.get_nowait()
            except queue.Empty:
                continue
            self.update(batch, update_step)

            update_step.value += 1
            if update_step.value % 1000 == 0:
                print("Training step ", update_step.value)

        ## training_on.value = 0
        empty_torch_queue(self.learner_queue)

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

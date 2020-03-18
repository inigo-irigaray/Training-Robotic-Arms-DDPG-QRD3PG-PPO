import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment




class D3PGAgent:
    def __init__(self, config, actor, global_episode, agent_id=0, explore=True):
        self.config = config
        self.actor = actor
        self.agent_id = agent_id
        self.explore = explore
        self.epsilon = self.config.epsilon

        self.local_episode = 0
        self.global_episode = global_episode

        self.env = UnityEnvironment(file_name=self.config.env)
        self.brain_name = self.env.brain_names[0]

    def _step(self, obs):
        action = self.actor(obs)
        action = action.data.cpu().numpy()
        if self.explore:
            action += self.epsilon * np.random.normal(size=action.shape)
        action = np.clip(action, -1, 1)
        return action

    def _update_actor_learner(self, train, learner_queue):
        if not train.value:
            return
        try:
            source = learner_queue.get_nowait()
        except:
            return
        target = self.actor
        for target_param, source_param in zip(target.parameters(), source):
            weights = torch.tensor(source_param).float()
            target_param.data.copy_(weights)
        del source

    def run(self, train, replay_queue, learner_queue, writer=None):
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        num_agents = len(env_info.agents) # number of agents within each agent in the engine
        episode = 0
        reward_100 = deque(maxlen=100)

        while train.value:
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            obs = env_info.vector_observations
            total_rewards = np.zeros(num_agents)
            t = time.time()
            it = 0
            self.local_episode += 1
            self.global_episode.value += 1
            while True:
                it += 1
                obs = torch.FloatTensor(np.vstack(obs))
                actions = self._step(obs)
                env_info = self.env.step(actions)[self.brain_name]
                next_obs = env_info.vector_observations
                rewards = env_info.rewards
                total_rewards += rewards
                dones = env_info.local_done

                try:
                    replay_queue.put_nowait([obs, actions, rewards, next_obs, dones])
                except:
                    pass

                if np.any(dones):
                    mean_reward = np.mean(total_rewards)
                    if writer is not None:
                        writer.add_scalar('mean_episode_reward', mean_reward, episode)
                    print("Agent %d, done local episode %d for an average\
                          reward of %.3f in %.2f seconds, iteration %d." %
                          (self.agent_id, episode, mean_reward, (time.time() - t), it))
                    t = time.time()
                    reward_100.append(mean_reward)
                    break
                    # return mean_reward

                obs = next_obs

            if self.explore==True:
                self._update_actor_learner(train, learner_queue)

            ### GET RID OF THIS IF I USE TRAIN.VALUE????
            episode += 1
            if np.mean(reward_100) >= 30.0:
                print("Solved the environment for agent %d in %d episodes and %.2f minutes."
                      % (self.agent_id, episode, (time.time() / 60)))
                self.env.close()
                break

        # Clears replay_queue
        while True:
            try:
                clearer = replay_queue.get_nowait()
                del clearer
            except:
                break
        replay_queue.close()

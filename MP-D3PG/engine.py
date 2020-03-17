import multiprocessing as mp
import time

import torch.multiprocessing as tmp
from tensorboardX import SummaryWriter

from agent import D3PGAgent
from buffer import ReplayBuffer
from learner import Learner




def sampler_worker(config, num_agents, training_on, replay_queue,
                   batch_queue, global_episode, update_step):
    """
    Function that transfers replay to the buffer and batches from buffer to the queue.
    """
    writer = SummaryWriter()

    # Creates replay buffer
    repbuffer = ReplayBuffer(capacity=config.capacity, num_agents=)

    while training_on.value:
        # Transfers transitions to replay buffer
        n = replay_queue.qsize()
        for _ in range(n):
            transition = replay_queue.get()
            repbuffer.add(*transition)

        # Transfers sample batch from buffer to the batch_queue
        if repbuffer.filled < config.batch_size:
            continue

        try:
            sample = repbuffer.sample(config.batch_size)
            batch_queue.put_nowait(sample)
        except:
            time.sleep(0.1)
            continue

        #### CHANGE CHANGE CHANGE !!!!!!!!!!!!!!!!
        step = update_step.value
        writer.scalar_summary("data_struct/global_episode", global_episode.value, step)
        writer.scalar_summary("data_struct/replay_queue", replay_queue.qsize(), step)
        writer.scalar_summary("data_struct/batch_queue", batch_queue.qsize(), step)
        writer.scalar_summary("data_struct/replay_buffer", len(replay_buffer), step)

    empty_torch_queue(batch_queue)




def learner_worker(config, obs_size, act_size, hid1, hid2, norm, actor,
                   tgt_actor, learner_queue, lr, learner_queue, gamma,
                   tau, training_on, batch_queue, update_step):
    learn = Learner(config, obs_size, act_size, hid1, hid2, norm, actor,
                    tgt_actor, learner_queue, lr, learner_queue, gamma, tau)
    learn.run(training_on, batch_queue, update_step)




def agent_worker(config, policy, learner_queue, global_episode, i, agent_type,
                 experiment_dir, training_on, replay_queue, update_step):
    agent = D3PGAgent(config,
                  policy=policy,
                  global_episode=global_episode,
                  n_agent=i,
                  agent_type=agent_type,
                  log_dir=experiment_dir)
    agent.run(training_on, replay_queue, learner_queue, update_step)




class Engine:
    def __init__(self, config):
        self.config = config

    def train():
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

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        batch_queue_size = self.config.batch_queue
        n_agents = self.config.num_agents

        # Data structures
        processes = []
        replay_queue = mp.Queue(maxsize=64)
        training_on = mp.Value('i', 1)
        update_step = mp.Value('i', 0)
        global_episode = mp.Value('i', 0)
        learner_w_queue = tmp.Queue(maxsize=n_agents)

        # Data sampler
        batch_queue = mp.Queue(maxsize=batch_queue_size)
        p = tmp.Process(target=sampler_worker,
                             args=(config, replay_queue, batch_queue, training_on,
                                   global_episode, update_step, experiment_dir))
        processes.append(p)

        # Learner (neural net training process)
        actor = Actor(obs_size, act_size, hid1=400, hid2=300, norm='layer')
        tgt_actor = TargetModel(actor)
        policy_net_cpu = PolicyNetwork(config['state_dim'], config['action_dim'],
                                          config['dense_size'], device=config['agent_device'])

        target_policy_net.share_memory()

        p = tmp.Process(target=learner_worker, args=(config, training_on, actor, tgt_actor, learner_queue,
                                                          batch_queue, update_step, experiment_dir))
        processes.append(p)

        # Single agent for exploitation
        p = tmp.Process(target=agent_worker, args=(config, tgt_actor, None, global_episode, 0, explore=False,
                                                   experiment_dir, training_on, replay_queue, update_step))
        processes.append(p)

        # Agents (exploration processes)
        for i in range(1, n_agents):
            p = tmp.Process(target=agent_worker,
                                 args=(config, actor.to('cpu'), learner_queue, global_episode, i, "exploration", experiment_dir,
                                       training_on, replay_queue, update_step))
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join()

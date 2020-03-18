import argparse
import multiprocessing as mp
import os
import time

import torch.multiprocessing as tmp
from tensorboardX import SummaryWriter

from agent import D3PGAgent
from buffer import ReplayBuffer
from learner import Learner




def sampler_worker(config, n_agents, train, replay_queue, batch_queue):
    """
    Function that transfers replay to the buffer and batches from buffer to the queue.
    """
    # Creates replay buffer
    repbuffer = ReplayBuffer(capacity=config.capacity, num_agents=n_agents)

    while train.value:
        # Transfers transitions to replay buffer
        n = replay_queue.qsize()
        for _ in range(n):
            transition = replay_queue.get() ### CHECK HERE FOR SOLVING ENV???
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

    # Clears batch_queue
    while True:
        try:
            clearer = batch_queue.get_nowait()
            del clearer
        except:
            break
    batch_queue.close()




def learner_worker(config, obs_size, act_size, hid1, hid2, norm, actor, tgt_actor,
                   lr, learner_queue, gamma, tau, batch_queue, update_step, train,
                   device='cpu', writer=None, filename=''):
    learn = Learner(config, obs_size, act_size, hid1, hid2, norm, actor,
                    tgt_actor, lr, learner_queue, gamma, tau)
    learn.run(batch_queue, update_step, train, device, writer, filename)




def agent_worker(config, actor, global_episode, agent_id=0, explore=True, writer=None):
    agent = D3PGAgent(config, actor, global_episode, agent_id, explore)
    agent.run(train, replay_queue, learner_queue, writer)




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
    saves_dir = run_dir / 'ddpg_robotic_arm.pt'

    writer = SummaryWriter(str(logs_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if torch.cuda.is_available() and self.config.cuda==True:
        device = 'gpu'
    else:
        device = 'cpu'

    batch_queue_size = config.batch_queue
    n_agents = config.n_agents

    # Communication tools across workers
    processes = []
    replay_queue = mp.Queue(maxsize=64)
    train = mp.Value('i', True)
    update_step = mp.Value('i', 0)
    global_episode = mp.Value('i', 0)
    learner_queue = tmp.Queue(maxsize=n_agents)
    batch_queue = mp.Queue(maxsize=batch_queue_size)

    env = UnityEnvironment(file_name=config.env)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    obs_size = env_info.vector_observations.shape[1]
    act_size = brain.vector_action_space_size

    # Data sampling process
    p = tmp.Process(target=sampler_worker,
                    args=(config, n_agents, train, replay_queue, batch_queue))
    processes.append(p)

    # Learner training process
    actor = Actor(obs_size, act_size, hid1=400, hid2=300, norm='layer')
    tgt_actor = TargetModel(actor)
    tgt_actor.share_memory()

    p = tmp.Process(target=learner_worker,
                    args=(config, obs_size, act_size, config.hid1, config.hid2, config.norm, actor,
                         tgt_actor, config.lr, learner_queue, config.gamma, config.tau, batch_queue,
                         update_step, train, device=device, writer=writer, filename=saves_dir))
    processes.append(p)

    # Single agent for exploitation
    p = tmp.Process(target=agent_worker,
                    args=(config, actor, global_episode, agent_id=0, explore=False, writer=writer))
    processes.append(p)

    # Agents' exploring process
    for i in range(1, n_agents):
        p = tmp.Process(target=agent_worker,
                        args=(config, actor.to('cpu'), config, actor, global_episode,
                              agent_id=i, explore=True, writer=writer))
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("End of all processes.")

if __name__ == "__main__":
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

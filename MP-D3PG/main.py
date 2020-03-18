import argparse
import multiprocessing as mp
import time

import torch.multiprocessing as tmp
from tensorboardX import SummaryWriter

from agent import D3PGAgent
from buffer import ReplayBuffer
from learner import Learner




def sampler_worker(config, num_in_agents, train, replay_queue, batch_queue):
    """
    Function that transfers replay to the buffer and batches from buffer to the queue.
    """
    # Creates replay buffer
    repbuffer = ReplayBuffer(capacity=config.capacity, num_agents=num_in_agents)

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
    agent = D3PGAgent(config, actor, global_episode, agent_id=0, explore=True)
    agent.run(train, replay_queue, learner_queue, writer=None)




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

    if torch.cuda.is_available() and self.config.cuda==True:
        cuda = True
    else:
        cuda = False

    batch_queue_size = config.batch_queue
    n_agents = config.num_agents

    # Communication tools across workers
    processes = []
    replay_queue = mp.Queue(maxsize=64)
    train = mp.Value('i', True)
    update_step = mp.Value('i', 0)
    global_episode = mp.Value('i', 0)
    learner_queue = tmp.Queue(maxsize=n_agents)
    batch_queue = mp.Queue(maxsize=batch_queue_size)

    # Data sampling process
    p = tmp.Process(target=sampler_worker,
                    args=(config, replay_queue, batch_queue, train,
                          global_episode, update_step, experiment_dir))
    processes.append(p)

    # Learner training process
    actor = Actor(obs_size, act_size, hid1=400, hid2=300, norm='layer')
    tgt_actor = TargetModel(actor)
    # policy_net_cpu = PolicyNetwork(config['state_dim'], config['action_dim'],
                                      #config['dense_size'], device=config['agent_device'])

    tgt_actor.share_memory()

    p = tmp.Process(target=learner_worker, args=(config, actor, tgt_actor, learner_queue, # training_on
                                                 batch_queue, update_step, experiment_dir))
    processes.append(p)

    # Single agent for exploitation
    p = tmp.Process(target=agent_worker, args=(config, tgt_actor, None, global_episode, 0, explore=False,
                                               experiment_dir, replay_queue, update_step)) # training_on
    processes.append(p)

    # Agents' exploring process
    for i in range(1, n_agents):
        p = tmp.Process(target=agent_worker,
                        args=(config, actor.to('cpu'), learner_queue, global_episode, i, "exploration", experiment_dir,
                                   replay_queue, update_step)) # training_on
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

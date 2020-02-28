# A Comparative Study of Distributed Methods Training Robotic Arms Using DDPG, D4PG, PPO & A-PPO with Multiple Agents (In Progress):

<p align=justify><b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(INITIAL DRAFT VERSION BOUND FOR UPDATES WITH FINAL RESULTS AS THE PROJECT ADVANCES) The aim of this project is to demonstrate the robustness of state-of-the-art deep reinforcement learning models (DRL) training robotic systems, e.g. robotic arms, as well as other challenges with continuous action spaces. Additionally, it experiments with deep distributed versions of DRL architectures, to test empirically the effects of parallelized learning and the advantages it can bring for industrial implementations of Deep Learning in the private sector as the field becomes increasingly essential to companies. It, specifically, implements Distributed Distributional Deep Deterministic Policy Gradients (D4PG) and an asynchronous version of Proximal Policy Optimization (A-PPO). At this point in the experiment, variants of the DDPG and D3PG algorithms have successfully completed training for robotic arms in the Unity Reacher environment, an ablation study of the impact of different components of the algorithm is underway, and the D4PG and A-PPO implementations are in progress and waiting for the results to update this abstract.</b></p>

-------

### Hyperparameters

#### DDPG (best performance with layer normalization)

    --env = '/data/Reacher_Linux_NoVis/Reacher.x86_64'
    --cuda = True
    --seed = 1
    --hid1 = 400
    --hid2 = 300
    --norm = 'layer'
    --lr = 1e-4
    --epsilon = 0.3
    --gamma = 0.99
    --tau = 1e-3
    --capacity = 100000
    --batch_size = 128
   
#### QR-D3PG (best performance with layer normalization)

    --env = '/data/Reacher_Linux_NoVis/Reacher.x86_64'
    --cuda = True
    --seed = 1
    --quant = 5
    --hid1 = 400
    --hid2 = 300
    --norm = 'layer'
    --lr = 1e-4
    --epsilon = 0.3
    --gamma = 0.99
    --tau = 1e-3
    --capacity = 100000
    --batch_size = 128

#### PPO (best performance with layer normalization)

    --env = '/data/Reacher_Linux_NoVis/Reacher.x86_64'
    --cuda = True
    --seed = 1
    --hid1 = 400
    --hid2 = 300
    --norm = 'layer'
    --lr = 1e-4
    --gamma = 0.99
    --gae_lambda = 0.95
    --epochs = 10
    --update_timestep = 375
    
---------

## 1. DDPG

<p align=justify>Deep Deterministic Policy Gradients (<b>DDPG</b>) is a policy gradients, model-free DRL method for continuous action spaces that learns off-policy to parameterize the deterministic target policy from an exploratory behaviour policy interacting with the environment. direction of gradients, critic bootstrapping etc.</p>
 
· Description of Deep Deterministic Policy Gradients based on research paper + specific implementation of the code

#### Policy Gradients: Actor-Critic

· Actor-critic target deterministic policy through gradient ascent learns from exploration online algorithm

#### Exploration: OUNoise vs RandomNoise

 · Conventional: Ornstein-Uhlenbeck Noise adds stochasticity and randomness to deterministic model for exploration
 
 · Random Noise: Adds simplicity and readability to the code, after empirical tests NO DETERIORATION of training --- Preferred choice
 
#### Normalization Layers

· NoNorm, BatchNorm, LayerNorm
 
#### PrioBuffer

· Does not blend well with the algorithm, adds significant computational burden and time cost for not radical improvement in performace episode-wise. Change discarded.

## 2. QR-D3PG

#### Distributional Approaches to Reinforcement Learning

. Why use them?

#### Quantile Regression Distributional Deep Deterministic Policy Gradients (QR-D3PG)

. Massive improvement.

#### Normalization Layers

· NoNorm, BatchNorm, LayerNorm

## 3. PPO

#### Policy, old policy, critic

· 

#### Generalized Advantage Estimation

· Implementation in this PPO version and effects

## 4. Comparative Analysis of Best-Performing Models

· 

## 5. Future Work

### 5.1 MP-D3PG

#### MultiProcessing Distributed Deep Deterministic Policy Gradients (MP-D3PG)

· APE-X

### 5.2 D4PG

#### Combining everything into one single algorithm

· Results

### 5.3 Asynchronous PPO

· Description of Proximal Policy Optimization and specifics of this implementation

## References

PAPERS:

D. Silver, G. Lever et al., 'Deterministic Policy Gradient Algorithms', 2014.

T.P. Lillicrap, J.J. Hunt et al., 'Continuous Control with Deep Reinforcement Learning', 2015.

G. Barth-Maron, M.W. Hoffman et al., 'Distributed Distributional Deterministic Policy Gradients', 2018.

M.G. Bellemare, W. Dabney et al., 'A DistributionalPerspective on Reinforcement Learning', 2017.

W. Dabney, M. Rowland, M.G. Bellemare & R. Munos, 'Distributional Reinforcement Learning with Quantile Regression', 2017.

W. Dabney, G. Ostrovski et al., 'Implicit Quantile Regression Networks for Distributional Reinforcement Learning', 2018.

PPO

GAE

S. Ioffe & C. Szegedy, 'Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift', 2015.

J.L. Ba, J.R. Kiros & G.E. Hinton, 'Layer Normalization', 2016.

T. Schaul, J. Quan, I. Antonoglou & D. Silver, 'Prioritized Experience Replay', 2016.

AcceleratedMethods4RL -- Ray implem

IMPALA: Scalable.....

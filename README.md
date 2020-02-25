# Comparative Distributed Robotic Arms Training Using DDPG, D4PG & A-PPO with 20 Agents (In Progress):

<p align=justify><b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(INITIAL DRAFT VERSION BOUND FOR UPDATES WITH FINAL RESULTS AS THE PROJECT ADVANCES) The aim of this project is to demonstrate the robustness of state-of-the-art deep reinforcement learning models (DRL) training robotic systems, e.g. robotic arms, as well as other challenges with continuous action spaces. Additionally, it experiments with deep distributed versions of DRL architectures, to test empirically the effects of parallelized learning and the advantages it can bring for industrial implementations of Deep Learning in the private sector as the field becomes increasingly essential to companies. It, specifically, implements Distributed Distributional Deep Deterministic Policy Gradients (D4PG) and an asynchronous version of Proximal Policy Optimization (A-PPO). At this point in the experiment, the DDPG algorithm has successfully completed training for robotic arms in the Unity Reacher environment, and the D4PG and A-PPO implementations are in progress and waiting for the results to update this abstract.</b></p>

-------
<p align=justify><b><a href=https://github.com/inigo-irigaray/Parallel-Training-Robotic-Arms-DDPG-D4PG-APPO/blob/master/README.md#1-preliminaries>1. Preliminaries.</a></b> Introduces a conceptual background on Deep Reinforcement Learning and the Reacher Unity Environment.</p>

<p align=justify><b><a href=https://github.com/inigo-irigaray/Parallel-Training-Robotic-Arms-DDPG-D4PG-APPO/blob/master/README.md#2-ddpg>2. DDPG.</a></b> Explains in detail the Deep Deterministic Policy Gradients algorithm, analyzes its performance training robotic arms for this environment, as well as some changes implemented on the baseline algorithm.</p>

<p align=justify><b><a href=https://github.com/inigo-irigaray/Parallel-Training-Robotic-Arms-DDPG-D4PG-APPO/blob/master/README.md#3-qr-d3pg>3. QR-D3PG.</a></b> </p>

<p align=justify><b><a href=https://github.com/inigo-irigaray/Parallel-Training-Robotic-Arms-DDPG-D4PG-APPO/blob/master/README.md#4-mp-d3pg>4. MP-D3PG.</a></b> </p>

<p align=justify><b><a href=https://github.com/inigo-irigaray/Parallel-Training-Robotic-Arms-DDPG-D4PG-APPO/blob/master/README.md#5-d4pg>5. D4PG.</a></b> Analyzes the limitations on DDPG and the improvements introduced to reach the Distributed Distributional Deep Deterministic Policy Gradients (D4PG) algorithm, as well as its performance on this environment.</p>

<p align=justify><b><a href=https://github.com/inigo-irigaray/Parallel-Training-Robotic-Arms-DDPG-D4PG-APPO/blob/master/README.md#6-ppo>6. PPO.</a></b> </p>

<p align=justify><b><a href=https://github.com/inigo-irigaray/Parallel-Training-Robotic-Arms-DDPG-D4PG-APPO/blob/master/README.md#7-asynchronous-ppo>7. A-PPO.</a></b> Implements an asynchronous version of the Proximal Policy Optimization (A-PPO) algorithm to evaluate the effectiveness of another state-of-the-art DRL method in a distributed setting against DDPG and D4PG.</p>

<p align=justify><b><a href=https://github.com/inigo-irigaray/Parallel-Training-Robotic-Arms-DDPG-D4PG-APPO/blob/master/README.md#8-a-comparative-analysis-of-the-models>8. Comparative Analysis.</a></b> Compares the strengths and performance of the three algorithms training robotic arms.</p>

<p align=justify><b><a href=https://github.com/inigo-irigaray/Parallel-Training-Robotic-Arms-DDPG-D4PG-APPO/blob/master/README.md#9-future-work>9. Future Work.</a></b> Explores potential avenues of interest for future experiments.</p>

---------
## 1. Preliminaries

### Environment

<p align=justify>In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.</p>

<p align=justify>The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.</p>

<p align=justify> The environment consists of 20 agents acting simultaneously, each corresponding to a robotic arm. The agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically, fter each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores. This yields an average score for each episode (where the average is over all 20 agents).</p>

<p align=justify>The environment can be installed from the following links:</p>
 
 · <a href=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip>Linux</a>
 
 · <a href=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip>MacOS</a>
 
 · <a href=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip>Windows(32-bit)</a>
 
 · <a href=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip>Windows(64-bit)</a>
 
 · <a href=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip>AWS(headless)</a>
 
 ### Requirements

    tensorflow==1.7.1
    Pillow>=4.2.1
    matplotlib
    numpy>=1.11.0
    jupyter
    pytest>=3.2.2
    docopt
    pyyaml
    protobuf==3.5.2
    grpcio==1.11.0
    torch==0.4.0
    pandas
    scipy
    ipykernel
    tensorboardX==1.4
    unityagents

## Running the models

<p align=justify> To run the different models available in this repository one only needs to clone/download from this repository the appropiate files from the folder of the model he/she wants to run and write in the command line: </p>

    $ python main.py
or

    $ python3 main.py
    
<p align=justify> which will start training the model from scratch until it reaches the environment's goal.</p>

## 2. DDPG

<p align=justify>Deep Deterministic Policy Gradients (<b>DDPG</p>) is a policy gradients, model-free DRL method for continuous action spaces that learns off-policy to parameterize the deterministic target policy from an exploratory behaviour policy interacting with the environment. direction of gradients, critic bootstrapping etc.</p>
 
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

## 3. QR-D3PG

#### Distributional Approaches to Reinforcement Learning

. Why use them?

#### Quantile Regression Distributional Deep Deterministic Policy Gradients (QR-D3PG)

. Massive improvement.

#### Normalization Layers

· NoNorm, BatchNorm, LayerNorm

## 4. MP-D3PG

#### Distributed Systems and Multiprocessing

· Overview of multiprocessing.

#### MultiProcessing Distributed Deep Deterministic Policy Gradients (MP-D3PG)

· Implementation and impact.

## 5. D4PG

#### Combining everything into one single algorithm

· Results

## 6. PPO

#### Policy, old policy, critic

· 

#### Generalized Advantage Estimation

· Implementation in this PPO version and effects

## 7. Asynchronous PPO

· Description of Proximal Policy Optimization and specifics of this implementation

#### Asynchronous PPO

· TO DO: Ray implementation --> empiric results compared to PPO (NOTE: not always positive on performance in other environments)

## 8. A Comparative Analysis of the Models

· 

## 9. Future Work

· SAC

## References

PAPERS:

DPG

DDPG

C51(ADistributionalPerspective.......)

QR

IQN

D4PG

PPO

GAE

BatchNorm

LayerNorm

PrioBuff

AcceleratedMethods4RL -- Ray implem

IMPALA: Scalable.....

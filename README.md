# Comparative Distributed Robotic Arms Training Using DDPG, D4PG & A-PPO with 20 Agents (In Progress):

<p align=justify><b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(INITIAL DRAFT VERSION BOUND FOR UPDATES WITH FINAL RESULTS AS THE PROJECT ADVANCES) The aim of this project is to demonstrate the robustness of state-of-the-art deep reinforcement learning models (DRL) training robotic systems, specifically robotic arms, as well as other challenges with continuous action spaces. Additionally, it experiments with deep distributed versions of DRL architectures, to test empirically the effects of parallelized learning and the advantages it can bring for industrial implementations of Deep Learning in the private sector as the field becomes increasingly essential to companies. Specifically, it implements Distributed Distributional Deep Deterministic Policy Gradients (D4PG) and an asynchronous version of Proximal Policy Optimization (A-PPO) with the Ray python framework for reinforcement learning and distributed applications. At this point in the experiment, the DDPG algorithm has successfully completed training for robotic arms in the Unity Reacher environment, and the D4PG and A-PPO implementations are in progress and waiting for the results to update this abstract.</b></p>

-------
<p align=justify><b><a href=https://github.com/inigo-irigaray/Parallel-Training-Robotic-Arms-DDPG-D4PG-APPO/blob/master/README.md#1-preliminaries>1. Preliminaries.</a></b> Introduces a conceptual background on Deep Reinforcement Learning and the Reacher Unity Environment.</p>

<p align=justify><b><a href=https://github.com/inigo-irigaray/Parallel-Training-Robotic-Arms-DDPG-D4PG-APPO/blob/master/README.md#2-ddpg>2. DDPG.</a></b> Explains in detail the Deep Deterministic Policy Gradients algorithm, analyzes its performance training robotic arms for this environment, as well as some changes implemented on the baseline algorithm.</p>

<p align=justify><b><a href=https://github.com/inigo-irigaray/Parallel-Training-Robotic-Arms-DDPG-D4PG-APPO/blob/master/README.md#3-d4pg>3. D4PG.</a></b> Analyzes the limitations on DDPG and the improvements introduced to reach the Distributed Distributional Deep Deterministic Policy Gradients (D4PG) algorithm, as well as its performance on this environment.</p>

<p align=justify><b><a href=https://github.com/inigo-irigaray/Parallel-Training-Robotic-Arms-DDPG-D4PG-APPO/blob/master/README.md#4-asynchronous-ppo>4. A-PPO.</a></b> Implements an asynchronous version of the Proximal Policy Optimization (A-PPO) algorithm to evaluate the effectiveness of another state-of-the-art DRL method in a distributed setting against DDPG and D4PG.</p>

<p align=justify><b><a href=https://github.com/inigo-irigaray/Parallel-Training-Robotic-Arms-DDPG-D4PG-APPO/blob/master/README.md#5-a-comparative-analysis-of-the-models>5. Comparative Analysis.</a></b> Compares the strengths and performance of the three algorithms training robotic arms.</p>

<p align=justify><b><a href=https://github.com/inigo-irigaray/Parallel-Training-Robotic-Arms-DDPG-D4PG-APPO/blob/master/README.md#6-future-work>6. Future Work.</a></b> Explores potential avenues of interest for future experiments.</p>

---------
## 1. Preliminaries

### Environment

<p align=justify>In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.</p>

<p align=justify>The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.</p>

<p align=justify> The environment consists of 20 agents acting simultaneously, each corresponding to a robotic arm. The agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically, fter each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores. This yields an average score for each episode (where the average is over all 20 agents).</p>

· Describe the environment

· Present an overview of DRL policy gradients

## 2. DDPG

· Description of Deep Deterministic Policy Gradients based on research paper + specific implementation of the code

#### Policy Gradients: Actor-Critic

· Actor-critic target deterministic policy through gradient ascent learns from exploration online algorithm

#### Exploration: OUNoise vs RandomNoise

 · Conventional: Ornstein-Uhlenbeck Noise adds stochasticity and randomness to deterministic model for exploration
 
 · Random Noise: Adds simplicity and readability to the code, after empirical tests NO DETERIORATION of training --- Preferred choice

## 3. D4PG

· Overview of weaknesses of DDPG implementation in the previous section and room for improvements tested and discussed in following points that lead to this project's D4PG implementation

#### PrioBuffer

· Minor relevant impact on a standalone basis, blends well with other improvements down the line to significantly impact the algorithms' performance

#### NoNorm vs BatchNorm vs LayerNorm

· BatchNorm not blending well with priobuffer

· LayerNorm on 1st layer of Actor & Critic -- remarkable positive impact wrt NoNorm. on more layers leads to overkill, strong deterioration of training

#### Multi-Training Every X t-steps

· Strong robust episode scores early-on BUT at the expense of time (running 10 training steps every 20 timesteps was prohibitively time-consuming, 4 training steps was better, but still takes a lot longer than 1 training step every timestep ----> hints at benefits of parallelization of learning==MORE learning + LESS time)

#### Distributional

· Implementation of C51 effects in progress

· TO DO: try quantile regression distribution.

· TO DO: Research Implicit Quantile Networks.

#### N-step

· Implement step unrolling with reward discounting --> has shown positive effects for other envs

#### Ray Parallelization

· TO DO implement ray rllib parallelization with remote actors tasks and learners and test empirically how well it blends with this particular environment

## 4. Asynchronous PPO

· Description of Proximal Policy Optimization and specifics of this implementation

#### Generalized Advantage Estimation

· Implementation in this PPO version and effects

#### Asynchronous PPO

· TO DO: Ray implementation --> empiric results compared to PPO (NOTE: not always positive on performance in other environments)

## 5. A Comparative Analysis of the Models

## 6. Future Work

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

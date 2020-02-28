## 1. Preliminaries

<p align=justify>This project is the second among three of Udacity's Deep Reinforcement Learning Nanodegree and aims to show the power of DRL methods to solve robotics and other continuous control problems. In particular, I have implemented different versions of Deterministic Policy Gradients to run an ablation study of the improvements that lead to Distributed Distributional Deep Deterministic Policy Gradients, as well as a Proximal Policy Optimization method to build upon it and implement an asynchronous variant of the baseline model. This project relies on a strong foundation based on Udacity's nanodegree, a thorough research into the DRL research literature, online lectures from top universities on the topic and the textbook Deep Reinforcement Learning Hands-On.</p>

<p align=justify><b>NOTE:</b> This initial deliverable is designed to meet Udacity's rubric criteria to pass the project while I work on the final implementation of Multiprocessing for D3PG and on improvements for PPO (mostly parameter tweaking), and prepare the presentation of a more detailed report this project and README.md in blog format for my personal portfolio. Therefore, MP-D3PG, D4PG and A-PPO are currently empty folders. All the others contain the necessary files for this implemntation, tensorboard data and some of the successful saved models after training.</p>

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
 
 Nanodegree's prerequisites: <a href=https://github.com/udacity/deep-reinforcement-learning/#dependencies>link.</a>
 
    python==3.6
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

### Running the models

<p align=justify> To run the different models available in this repository one only needs to clone/download from this repository the appropiate files from the folder of the model he/she wants to run and write in the command line: </p>

    $ python main.py
or

    $ python3 main.py
    
<p align=justify> which will start training the model from scratch until it reaches the environment's goal.</p>

<p align=justify>For example:
<p align=justify>1. Clone this repository.</p>
<p align=justify>2. Install all required dependencies.</p>
<p align=justify>3. And run the the command:</p>
     
    $ python3 path_to_DDPG_folder/main.py --cuda=True

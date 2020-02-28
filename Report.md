# An Ablation Study of Policy Gradients Algorithms for Continuous Control with Multiple Agents

<p align=justify><b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The aim of this project is to demonstrate the robustness of state-of-the-art deep reinforcement learning models (DRL) for continuous control and training robotic systems, e.g. robotic arms, as its potential benefits for industrial and professional applications become increasingly noticeable. It demonstrates the strength of quantile regression distributional approaches to reinforcement learning, completing the task by episode 105 (just above the minimum required runs by the task of 100 episodes) and potentially achieving a new state-of-the-art result for the Udacity project (needs to be confirmed). Additionally, it performs an ablation study of multiple variations of Deterministic Policy Gradients and Proximal Policy Optimization, which highlight the importance of normalization layers and the representational advantage of approximating value distributions through distributional RL, instead of estimating the mean value function. The next goal is to implement distributed versions of the aforementioned algorithms (MP-D3PG, D4PG & A-PPO) that could potentially yield significant advantages for practical industry applications.</b></p>

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

#### PPO (best performance without normalization layers)

    --env = '/data/Reacher_Linux_NoVis/Reacher.x86_64'
    --cuda = True
    --seed = 1
    --hid1 = 400
    --hid2 = 300
    --norm = None
    --lr = 1e-4
    --gamma = 0.99
    --gae_lambda = 0.95
    --epochs = 10
    --update_timestep = 375
    
---------
<p align=justify><sub>NOTE: all implementations include utility functions prep_training and prep_rollouts to switch between GPUs and CPUs when necessary, and save, init_from_env, init_from_save to save trained model parameters and initialize the environment from scratch or from a file of saved trained weights.</sub></p>

## 1. DDPG

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Deep Deterministic Policy Gradients (<b>DDPG</b>) is a policy gradients, model-free, off-policy DRL method for continuous action spaces. It learns from an exploratory behaviour policy to parameterize the deterministic target policy maximixing state-action values. The online policy interacts with the environment to generate batches of experience that accumulate state, action, reward, next_state and done info transitions in an experience replay buffer. The experiences are then sampled randomly from the buffer to perform mini-batch stochastic gradient ascent of the policy loss, which is equal to the expected Q-value (since our goal is to learn the policy that maximizes it). This Q-value is in turn estimated from a critic network that models the value function, is differentiable and is learned by performing stochastic gradient descent over the mean squared temporal-difference error (MSTDE).</p>
 
#### Actor-Critic

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The <b>Actor</b> network that parameterizes policies consists of <b>three fully-connected layers</b> of shape (33, 400), (400, 300), (300, 4), with <b>Rectified Linear Unit non-linearities in-between</b>, and a <b>Tanh non-linearity</b> to output a float within the [-1., 1.] range that this environment accepts for the action space. The <b>Critic</b> network that approximates the value function is defined by a sequence of <b>three fully-connected layers</b> of shape (33, 400), (404, 300), (300,1), with <b>two ReLU layers in-between</b>, and outputs the estimated mean value. Both networks parameterize the state space to understand the agents situation in the environment, which is why they take inputs of size 33, equal to the state dimension. Additionally, the critic network takes as input the action space in the second linear layer in order to calculate state-action values, which explains the difference between the output of the first layer (400) and the input of the second layer (404 = 400 + 4_action dim).</p>

#### Exploration: OUNoise vs RandomNoise

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;One of the key challenges in Reinforcement Learning is to find the right balance between optimization of the current policy being pursuit and <b>exploration</b> of alternatives. It is essential to induce a level of 'curiosity' in our agents to try different possibilities, since it is extremely unlikely to land in the path to the optimal policy from random initialization and subsequent progressive improvements through backpropagation. A level of randomness, then, becomes indispensable.</p>

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;The classical approach for DDPG has been to introduce <b>Ornstein-Uhlenbeck noise</b> for action selection. OU Noise generates noise as a paritcular function of a Gaussian distribution, which is then added to the output of the actor network for exploration. An alternative approach that was also tested for this experiment was the minimal addition of one <b>Noisy Linear layer</b>, which is a modification on the default linear fully-connected layer from PyTorch, before Tanh. The noise is, thus, parametized and learned. However, it empirically proved to be to aggresive at initialization and added computational and time cost. Alternatively, a third option proved to be cheap to implement, both computational and time-wise, and much more readable and neat for the simplicity of the code. This approach was simply adding <b>independent Gaussian noise</b> to the output of the actor, which was easily implemented in one line of code in the step() function with numpy.</p>
 
#### Normalization Layers

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;I then progressed to study the effects of normalization of inputs within the Actor and Critic networks in overall performance. <b>Batch and Layer normalization</b> have been widely incorporated into deep learning algorithms to allow for aggresive learning rates and faster training by standardizing the interlayer input and reducing internal covariate shift. Additionally, they can act as regularizers to improve generalization.</p>

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;However, after testing both types of normalization within the network architectures, they proved to damage learning when implemented in-between layers. However, they did manage to accelerate training when they were only introduced as the first layer of the network, even before the initial linear layer. Normalization, therefore, appears to ease training due to its regularizing effect and not by means of internal covariate shift reduction. However, this only partially true. While they both <b>reduced the number of episodes</b> to achieve the goal of a reward of over 30.0 for the past 100 episodes, from 127 in the baseline implementation to 125 for both batch and layer normalization, they <b>took longer to train</b> due to increased computations.</p>

<p align=justify>&nbsp;&nbsp;&nbsp;This suggests that adding these extra layers to the architecture design depends very much on the specifics of the application. The user will have to trade off the episodic cost of interacting with the environment (which can be very high for some physical robotics applications, like crashing self-driving cars) for computational and time cost (which may be more important for some applications that do not require physical hardware, for example). It is, therefore, left for the user as a hyperparameter choice (--norm = [None, batch, layer]).</p>

<p align=center><img src=https://github.com/inigo-irigaray/Training-Robotic-Arms-DDPG-QRD3PG-PPO/blob/master/imgs/DDPG/episode_reward_step.png height=330 width=650></p>

<p align=center><sub>Episode reward per number of episodes. Baseline (red), Batch (blue), Layer (Orange)</sub></p>

<p align=center><img src=https://github.com/inigo-irigaray/Training-Robotic-Arms-DDPG-QRD3PG-PPO/blob/master/imgs/DDPG/episode_reward_time.png height=330 width=650></p>

<p align=center><sub>Episode reward per training time. Baseline (red), Batch (blue), Layer (Orange)</sub></p>

<p align=center><img src=https://github.com/inigo-irigaray/Training-Robotic-Arms-DDPG-QRD3PG-PPO/blob/master/imgs/DDPG/speed.png height=330 width=650></p>

<p align=center><sub>Speed (frames per second). Baseline (red), Batch (blue), Layer (Orange)</sub></p>

#### PrioBuffer

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;I finally implemented a Prioritized Experience Replay Buffer. However, since performance was already pretty robust at 125 episodes, there was not much room improvement and it added signifant overhead costs for computation. Additionally, it did not blend well with my batch and layer normalization implementations. It was, therefore, discarded as the simpler baseline experience replay was faster. To implement PER you will need to install OpenAI's baselines as a prerequisite, since I imported directly the Tree Search algorithms from there. For basic ER, it is not required.</p>

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

J.L. Ba, J.R. Kiros & G.E. Hinton, 'Layer Normalization', 2016.

G. Barth-Maron, M.W. Hoffman et al., 'Distributed Distributional Deterministic Policy Gradients', 2018.

M.G. Bellemare, W. Dabney et al., 'A DistributionalPerspective on Reinforcement Learning', 2017.

W. Dabney, G. Ostrovski et al., 'Implicit Quantile Regression Networks for Distributional Reinforcement Learning', 2018.

W. Dabney, M. Rowland, M.G. Bellemare & R. Munos, 'Distributional Reinforcement Learning with Quantile Regression', 2017.

L. Espeholt, H. Soyer, R. Munos et al., 'IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures', 2018.

D. Horgan, J. Quan, D. Budden, G. Barth-Maron, M. Hessel, H. van Hasselt & D. Silver, 'Distributed Prioritized Experience Replay', 2018.

S. Ioffe & C. Szegedy, 'Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift', 2015.

T.P. Lillicrap, J.J. Hunt et al., 'Continuous Control with Deep Reinforcement Learning', 2015.

T. Schaul, J. Quan, I. Antonoglou & D. Silver, 'Prioritized Experience Replay', 2016.

J. Schulman, P. Moritz, S. Levine, M.I. Jordan & P. Abbeel, 'High-Dimensional Continuous Control Using Generalized Advantage Estimation', 2015.

J. Schulman, F. Wolski, P. Dhariwal, A. Radford & O. Klimov, 'Proximal Policy Optimization Algorithms', 2017.

D. Silver, G. Lever et al., 'Deterministic Policy Gradient Algorithms', 2014.

A. Stooke & P. Abbeel, 'Accelerated Methods for Deep Reinforcement Learning', 2018.

# Comparative Distributed Robotic Arms Training Using DDPG, D4PG & A-PPO with 20 Agents (In Progress):

<p align=justify><b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(INITIAL DRAFT VERSION BOUND FOR IMPROVEMENTS) This project empirically shows the benefits of combining Deep Reinforcement Learning (DLR) methods with popular Natural Language Processing (NLP) algorithms in the pursuit of state-of-the-art results in dialogue systems and other human language comprehension tasks. The experiment is based on the simple Cornell University Movie Dialogs database and integrates the sequence-to-sequence (seq2seq) model of LSTM networks into cross-entropy learning for pretraining and into the REINFORCE method. Thus, the algorithm leverages the power of stochasticity  inherent to Policy Gradient (PG) models and directly optimizes the BLEU score, while avoiding getting the agent stuck through transfer learning of log-likelihood training. This combination results in improved quality and generalization of NLP models and opens the way for stronger algorithms for various tasks, including those outside the human language domain.</b></p>

-------
<p align=justify><b><a href=https://github.com/inigo-irigaray/NLP-seq2seq-with-DeepRL#1-preliminaries>1. Preliminaries.</a></b> Introduces a conceptual background on Deep Reinforcement Learning and the Reacher Unity Environment.</p>

<p align=justify><b><a href=https://github.com/inigo-irigaray/NLP-seq2seq-with-DeepRL#2-seq2seq-with-cross-entropy--reinforce>2. DDPG.</a></b> Explains in detail the Deep Deterministic Policy Gradients algorithm, analyzes its performance training robotic arms for this environment, as well as some changes implemented on the baseline algorithm.</p>

<p align=justify><b><a href=https://github.com/inigo-irigaray/NLP-seq2seq-with-DeepRL#3-training--results>3. D4PG.</a></b> Analyzes the limitations on DDPG and the improvements introduced to reach the Distributed Distributional Deep Deterministic Policy Gradients (D4PG) algorithm, as well as its performance on this environment.</p>

<p align=justify><b><a href=https://github.com/inigo-irigaray/NLP-seq2seq-with-DeepRL#4-future-work>4. A-PPO.</a></b> Explores potential avenues of interest for future experiments.</p>


---------
## 1. Preliminaries

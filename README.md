# Comparative Distributed Robotic Arms Training Using DDPG, D4PG & PPO with 20 Agents (In Progress):

<p align=justify><b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This project empirically shows the benefits of combining Deep Reinforcement Learning (DLR) methods with popular Natural Language Processing (NLP) algorithms in the pursuit of state-of-the-art results in dialogue systems and other human language comprehension tasks. The experiment is based on the simple Cornell University Movie Dialogs database and integrates the sequence-to-sequence (seq2seq) model of LSTM networks into cross-entropy learning for pretraining and into the REINFORCE method. Thus, the algorithm leverages the power of stochasticity  inherent to Policy Gradient (PG) models and directly optimizes the BLEU score, while avoiding getting the agent stuck through transfer learning of log-likelihood training. This combination results in improved quality and generalization of NLP models and opens the way for stronger algorithms for various tasks, including those outside the human language domain.</b></p>

-------
<p align=justify><b><a href=https://github.com/inigo-irigaray/NLP-seq2seq-with-DeepRL#1-preliminaries>1. Preliminaries.</a></b> Introduces a conceptual background on the NLP literature and state-of-the-art algorithms for conversational modelling, machine translation and other key challenges in the field.</p>

<b><a href=https://github.com/inigo-irigaray/NLP-seq2seq-with-DeepRL#2-seq2seq-with-cross-entropy--reinforce>2. seq2seq with Cross-Entropy & REINFORCE - the algorithms.</a></b> Details the specifics of the algorithms used for this particular experiment and the core structure of the approximation models employed.</p>

<b><a href=https://github.com/inigo-irigaray/NLP-seq2seq-with-DeepRL#3-training--results>3. Training & Tests Discussion.</a></b> Analyzes the progress of the two different training methods until halting, and the corresponding performance of the model on the tests.</p>

<b><a href=https://github.com/inigo-irigaray/NLP-seq2seq-with-DeepRL#4-future-work>4. Future work.</a></b> Explores potential avenues of interest for future experiments.</p>


---------
## 1. Preliminaries

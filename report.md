# Solving the Unity ML Banana environment using Deep Q Learning

## Part I: The agent

The agent decides its actions using a Deep Neural Network evaluating the **action values** as a function of the **state** given by the environment. Its weights are updated using the **rewards** transmitted by the environment as responses to the actions. The Neural Network architecture is as follow:

Input: state vector (37 float values)
Input -> First layer (64 nodes) -> Leaky Relu (negative part slope = 0.01)
      -> Batch normalization
      -> Second layer (64 nodes) -> Leaky Relu (negative part slope = 0.01)
      -> Batch normalization
      -> Output layer (4 actions)

Batch normalization has been introduced to help the network optimization by centering the current node values around zero (See Pytorch documentation for example)

The leaky relu activation function has been chosen for its non-zero gradient in the negative part. No evidence of improvement has been seen, however I suspect some help in the optimization stability.

A **target copy** of this architectures is used to evaluate the **loss function** during optimization following the **Double Q-Learning** principle (https://arxiv.org/pdf/1509.06461.pdf). The target network weights (Wtarget) are updated softly using the local weights (Wlocal) of the main network (deciding the actions) by taking the barycenter ponderated by a factor **TAU**, taken to 1e-3 for this work:

Wtarget_new = TAU * Wlocal + (1 - TAU) Wtarget_old

The local network weights are updated **every 4 time steps** as follow:
1/ A **sample of sequences** State-Action-Reward-Next State-Done (**SARS+Done**) is taken from the past experience, stored in a **replay buffer** (we fixed a batch size of 64exemples)
2/ For each example, the **action-value function Q(s,.)** is estimated for the current state s using the **local network**
3/ For each example, the **expected reward** Q'(s) is estimated from the **action-value function Q(s',.)**  for the next state s' using the **target network** inserted in the **Bellman formula** Q'(s) = R + Gamma*Q(s',.). Where **Gamma** is the discount factor for the expected reward, set to 0.99.
4/ We use Q(s,_) from 2/ compared to Q'(s,_) from 3/ to optimize the weights of the local deep neural network, using an **Adam** optimizer with learning rate set to 1e-3.

The experience is stored in a **replay buffer** of maximal size of 1e5 examples.

## Part II: Training phase

To **train the agent**, we make it play episodes to make it learn from its experiences through the process described in part I. For each episode, we first **reset the environment** to obtain an initial state, then we repeat the following loop:
- chose an action and send it to the environment,
- store the **new_state** and **reward** received by the environment in variables,
- add the reward to the **score** of the current episode for performance tracking,
- replace the reward by a **rectified reward** (myReward) to add a penalty (set to -0.05) for actions leading to steps without collecting a banana,
- set the **new_state** to current state,
- **repeat** or **exit** the loop if the environment indicates the end of the episode.



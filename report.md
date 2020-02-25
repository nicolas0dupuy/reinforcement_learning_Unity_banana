# Solving the Unity ML Banana environment using Deep Q Learning

## Part I: The agent

### A) Learning algorithm

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

Warget_new = TAU * Wlocal + (1 - TAU) Wtarget_old

The local network weights are updated **every 4 time steps** as follow.

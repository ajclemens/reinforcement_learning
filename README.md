# Reinforcement Learning

by Anthony Clemens

I am using openAI `gym` to train reinforcement learning models to solve the cartpole control problem. I have implemented two solutions to the problem, a Deep Q-Network (DQN) algorithm using `keras`, and a neuroevolution of augmenting topologies (NEAT) genetic algorithm using `neat-python`.

![](./assets/cartpole_demo.gif)

**Figure 1:** Results for DQN agent after training on 2000 episodes. When I say agent I am describing the AI that interacts with the simulated environment. The resulting agent has effectively solved the problem at hand (it is able to balance the pole forever) without being explicitly told how to behave. Instead, simply observing the rewards of random actions was enough for the algorithm to construct a successful policy (strategy) to the problem at hand. Due to the fact the policy was not explicitly specified and obtained via random exploration this is denoted off-policy learning.

An essential concept to reinforcement learning is Q-learning, which is based upon the idea of a Q-function: 

> *The Q-function quantifies the reward an agent may recieve given it's current state and next action*. 

 Depending upon the complexity of the problem, it may be too difficult to write the explicit mathematical formula for the Q-function by hand. The Deep Q-Network algorithm solves this problem as a neural network is used to approximate the Q-function. In effect, the explicit mathematical formula for the Q-function is not needed.

 In simplest terms, the neural network maps the current state the agent is in to the next action with greatest reward. For the cartpole problem the state includes 4 variables, the cart position, cart velocity, pole angle, and pole angular velocity. Additionally, the two actions that can be taken are move the cart either left or right.

talk about Exploration rate
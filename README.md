# reinforcement_learning

I am using openAI gym to train reinforcement learning models to solve the cartpole control problem. I have implemented two solutions to the problem, a Deep Q-Network (DQN) algorithm using `keras`, and a neuroevolution of augmenting topologies (NEAT) genetic algorithm using `neat-python`.

An essential concept to reinforcement learning is Q-learning, which is based upon the idea of a Q-function: 

*quantifies the reward an agent may recieve given it's current state and next action*. 

When I say agent I am describing the AI that interacts with the simulated environment. Depending upon the problem, it may be too difficult to  write the explicit mathematical formula for the Q-function by hand. As the number of possible states and actions increase i.e. as the complexity of the problem increases this becomes increasingly difficult. The Deep Q-Network algoirthm uses a neural network to approximate this Q-function, so we don't need to explicitly write out the formula. 
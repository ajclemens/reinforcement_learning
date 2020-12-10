# source: https://keon.github.io/deep-q-learning/
# imports
import numpy as np
import random
from collections import deque  
import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# deep Q-learning network class
class DQN: 
    def __init__(self, env):
        # hyper parameters
        self.observation_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.learning_rate = 1e-3
        self.model = self.build_model()
        self.gamma = 0.95
        self.epsilon = 1 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=2000)

    # build keras model
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.observation_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # inputs state, returns action. Either random or models prediction
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    # memory of simulation episode for training
    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # train model
    def train_model(self, batch_size):
        # create minibatch
        minibatch = random.sample(self.memory, batch_size)
        # iterate through minibatch
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # calculate target (future discounted reward)
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0]) # future discounted reward

            target_f = self.model.predict(state)
            # setting the current action to the future discounted reward
            # the goal of this is to associate the better move with better reward
            target_f[0][action] = target 
            # fit model
            self.model.fit(state, target_f, epochs=1, verbose=0)
        # decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# main
if __name__ == '__main__':
    # make gym environment
    env = gym.make('CartPole-v0')
    # initialize DQN agent
    agent = DQN(env)

    # batch size
    batch_size = 32
    # number of episodes
    episodes = 2000
    for e in range(episodes):
        # reset in the beginning of each game
        state = env.reset()
        # we have to reshape
        state = np.reshape(state, [1, 4])

        # t represents each frame of the game
        tspan = 500
        for t in range(tspan):
            # render game
            env.render()
            # decide action
            action = agent.act(state)

            # advance the game one frame
            # reward is 1 for every frame survived
            next_state, reward, done, _ = env.step(action)
            # reshape
            next_state = np.reshape(next_state, [1, 4])
            # memorize
            agent.memorize(state, action, reward, next_state, done)
            # update state
            state = next_state

            if done:
                # print the score and break out of the loop
                print(f'episode: {e+1}/{episodes}, score: {t}')
                break
        # sometimes the first episode does not last long enough you don't have enough gameplay for one batch size.
        if len(agent.memory) > batch_size:
            agent.train_model(batch_size)
        else:
            print('Memories did not exceed batch size')
    # save model
    agent.model.save('./DSI/reinforcement_learning/assets/model')
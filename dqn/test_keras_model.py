# imports
import numpy as np
import gym
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# main
if __name__ == '__main__':
    # load model
    model = load_model('./DSI/reinforcement_learning/assets/model')
    # plot model
    plot_model(model, to_file='./DSI/reinforcement_learning/assets/keras_model.png', show_shapes=True)
    # create gym environment
    env = gym.make('CartPole-v0')
    # reset environment
    state = env.reset()
    # we have to reshape
    state = np.reshape(state, [1, 4])

    # loop through game
    tspan = 200
    for t in range(tspan):
        # render
        env.render()
        # get models action
        action = np.argmax(model.predict(state)[0])
        # step
        next_state, reward, done, _ = env.step(action)
        # if done break out
        if done:
            break
        # we have to reshape
        next_state = np.reshape(next_state, [1, 4])
        # update state
        state = next_state
        # print
        print(f'Time Survived: {t}')


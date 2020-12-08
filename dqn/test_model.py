import numpy as np
import gym
from tensorflow.keras.models import load_model


if __name__ == '__main__':
    # load model
    model = load_model('./DSI/reinforcement_learning/assets/model')
    # create gym environment
    env = gym.make('CartPole-v0')
    state = env.reset()
    state = np.reshape(state, [1, 4])

    tspan = 1000
    for t in range(tspan):
        env.render()
        action = np.argmax(model.predict(state)[0])
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        state = next_state
        print(f'Time Survived: {t}')


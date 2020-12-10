# imports
import numpy as np
import gym
import neat
import pickle
from visualize import draw_net

# main
if __name__ == '__main__':
    # initialize config object
    config = neat.config.Config(genome_type=neat.genome.DefaultGenome, reproduction_type=neat.reproduction.DefaultReproduction, species_set_type=neat.species.DefaultSpeciesSet, stagnation_type=neat.stagnation.DefaultStagnation, filename='./DSI/reinforcement_learning/neat/neat_config.txt')
    # load model
    with open('./DSI/reinforcement_learning/assets/neat_winner.pkl', 'rb') as f:
        winner = pickle.load(f)
    # draw neural network png
    draw_net(config, winner, view=False, filename='./DSI/reinforcement_learning/assets/neat_model', fmt='png')
    # create neural network
    net = neat.nn.feed_forward.FeedForwardNetwork.create(winner, config)
    # create gym environment
    env = gym.make('CartPole-v0')
    # reset game
    state = env.reset()

    # loop through game
    tspan = 200
    for t in range(tspan):
        # render
        env.render()
        # get models outputs
        outputs = net.activate(state)
        # choose action with greatest reward
        action = np.argmax(outputs)
        # step
        next_state, reward, done, _ = env.step(action)
        # if done break out
        if done:
            break
        # update state
        state = next_state
        # print
        print(f'Time Survived: {t}')


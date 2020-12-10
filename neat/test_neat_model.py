import numpy as np
import gym
import neat
import pickle
from visualize import draw_net
if __name__ == '__main__':
    config = neat.config.Config(genome_type=neat.genome.DefaultGenome, reproduction_type=neat.reproduction.DefaultReproduction, species_set_type=neat.species.DefaultSpeciesSet, stagnation_type=neat.stagnation.DefaultStagnation, filename='./DSI/reinforcement_learning/neat/neat_config.txt')
    # load model
    with open('./DSI/reinforcement_learning/assets/neat_winner.pkl', 'rb') as f:
        winner = pickle.load(f)
    draw_net(config, winner, view=False, filename='./DSI/reinforcement_learning/assets/neat_model', fmt='png')
    net = neat.nn.feed_forward.FeedForwardNetwork.create(winner, config)
    # create gym environment
    env = gym.make('CartPole-v0')
    state = env.reset()
    state = np.reshape(state, [1, 4])

    tspan = 1000
    for t in range(tspan):
        inputs = env.reset()
        env.render()
        outputs = net.activate(inputs)
        action = np.argmax(outputs)
        inputs, reward, done, _ = env.step(action)
        env.render()
        print(f'Time Survived: {t}')


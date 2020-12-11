# source: https://github.com/HackerShackOfficial/OpenAI-NEAT
# imports
import numpy as np
import gym
import neat
import sys
import pickle

# render flag, 0 off, 1 on, 0 trains faster
render_flag = 0
# number of generations
num_generations = 10000
# number of episodes per genome
num_episodes = 10
# max steps per episode
max_steps = 200

# input neural network, environment, number of episodes, 
# number of steps, render flag, and outputs average fitness for that neural network (genome)
def simulate_species(net, env, episodes, steps, render=False):
    fitnesses = []
    for runs in range(episodes):
        inputs = env.reset()
        total_reward = 0.0
        for j in range(steps):
            outputs = net.activate(inputs)
            action = np.argmax(outputs)
            inputs, reward, done, _ = env.step(action)
            if render_flag:
                env.render()
            if done:
                break
            total_reward += reward

        fitnesses.append(total_reward)

    fitness = np.array(fitnesses).mean()
    print("Species fitness: %s" % str(fitness))
    return fitness

# evaluate a genome
def eval_genome(genome, config):
    # create neural network
    net = neat.nn.feed_forward.FeedForwardNetwork.create(genome[1], config)
    return simulate_species(net, env, num_episodes, max_steps, render=render_flag)

# assigns fitness to genome
def eval_fitness(genomes, config):
    for genome in genomes:
        fitness = eval_genome(genome, config)
        genome[1].fitness = fitness

# train 
def train(env):
    # initialize config object
    config = neat.config.Config(genome_type=neat.genome.DefaultGenome, reproduction_type=neat.reproduction.DefaultReproduction, species_set_type=neat.species.DefaultSpeciesSet, stagnation_type=neat.stagnation.DefaultStagnation, filename='./DSI/reinforcement_learning/neat/neat_config.txt')
    # initialize pop object
    pop = neat.population.Population(config)
    # initialize stats object
    stats = neat.statistics.StatisticsReporter()
    # add stats object to pop object
    pop.add_reporter(stats)
    # run method
    pop.run(eval_fitness, num_generations)
    # winner
    winner = stats.best_genome()
    
    # save winner
    with open('./DSI/reinforcement_learning/assets/neat_winner.pkl', 'wb') as f:
       pickle.dump(winner, f)

    # print winner
    print(f'Best genome:\n{winner}')

# main
if __name__ == '__main__':
    # make environment
    env = gym.make('CartPole-v0')
    # train
    train(env)
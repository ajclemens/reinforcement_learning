# source: https://github.com/HackerShackOfficial/OpenAI-NEAT
# imports
import numpy as np
import gym
import neat
import sys
import pickle

render_flag = 0
num_generations = 10000
num_episodes = 10
max_steps = 200

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

def eval_genome(genome, config):
    print(genome)
    net = neat.nn.feed_forward.FeedForwardNetwork.create(genome[1], config)
    return simulate_species(net, env, num_episodes, max_steps, render=render_flag)

def eval_fitness(genomes, config):
    for genome in genomes:
        fitness = eval_genome(genome, config)
        genome[1].fitness = fitness

def train_network(env):
    config = neat.config.Config(genome_type=neat.genome.DefaultGenome, reproduction_type=neat.reproduction.DefaultReproduction, species_set_type=neat.species.DefaultSpeciesSet, stagnation_type=neat.stagnation.DefaultStagnation, filename='./DSI/reinforcement_learning/neat/neat_config.txt')
    pop = neat.population.Population(config)
    stats = neat.statistics.StatisticsReporter()
    pop.add_reporter(stats)

    pop.run(eval_fitness, num_generations)
    winner = stats.best_genome()
    
    with open('./DSI/reinforcement_learning/assets/neat_winner.pkl', 'wb') as f:
       pickle.dump(winner, f)

    print(f'Best genome:\n{winner}')

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    train_network(env)
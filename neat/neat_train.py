# source: https://github.com/HackerShackOfficial/OpenAI-NEAT
# imports
import numpy as np
import gym
import neat
import sys
import pickle

render_flag = 1
num_generations = 2
num_cores = 2 # # of cpu cores for parallel run
num_episodes = 1
max_steps = 1000
episodes = 1

def simulate_species(net, env, episodes=1, steps=5000, render=False):
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
        net = neat.nn.feed_forward.FeedForwardNetwork.create(genome[1], config)
        return simulate_species(net, env, num_episodes, max_steps, render=render_flag)

def eval_fitness(genomes, config):
    for genome in genomes:
        fitness = eval_genome(genome, config)
        genome[1].fitness = fitness

def worker_evaluate_genome(genome, config):
    net = neat.nn.feed_forward.FeedForwardNetwork(genome, config)
    return simulate_species(net, env, num_episodes, max_steps, render=render_flag)

def train_network(env):
    config = neat.config.Config(genome_type=neat.genome.DefaultGenome, reproduction_type=neat.reproduction.DefaultReproduction, species_set_type=neat.species.DefaultSpeciesSet, stagnation_type=neat.stagnation.DefaultStagnation, filename='./DSI/capstone/neat_config.txt')
    pop = neat.population.Population(config)
    stats = neat.statistics.StatisticsReporter()
    pop.add_reporter(stats)

    if render_flag:
        pop.run(eval_fitness, num_generations)
    else:
        pe = neat.parallel.ParallelEvaluator(num_workers=num_cores, eval_function=worker_evaluate_genome)
        winner = pop.run(pe.evaluate)
    

    winner = stats.best_genome()

    with open('./DSI/reinforcement_learning/assets/neat_winner.pkl', 'wb') as output:
       pickle.dump(winner, output)

    print(f'Best genome:\n{winner}')


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    train_network(env)
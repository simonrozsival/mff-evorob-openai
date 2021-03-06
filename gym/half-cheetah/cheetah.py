import os
import gym
import neat
import numpy as np
import visualize

env = gym.make('HalfCheetah-v1')
env = gym.wrappers.Monitor(env, 'results', force=True)
env.reset()

print("action space: {0!r}".format(env.action_space))
print(env.action_space.sample())
print("observation space: {0!r}".format(env.observation_space))


def evaluate_nn(net, visualise=False):
    fitness = 0
    observation = env.reset()
    for t in range(1000):
        if visualise:
            env.render()

        action = net.activate(observation)
        observation, reward, done, info = env.step(action)
        fitness += reward
        if done:
            break

    return fitness


def evaluate_genomes(genomes, config):
    for gid, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = evaluate_nn(net, visualise=False)


def run(config_path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)

    # setup some output statistics
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every 25 generations or 900 seconds.
    pop.add_reporter(neat.Checkpointer(25, 900))

    winner = pop.run(evaluate_genomes, 1500)
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    while True:
        evaluate_nn(net, visualise=True)

    # visualise the results
    visualize.plot_stats(stats, ylog=False, view=False, filename="fitness.svg")


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-cheetah')
    run(config_path)

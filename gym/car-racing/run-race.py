import os
import gym
import neat
import numpy as np

env = gym.make('CarRacing-v0')
env = gym.wrappers.Monitor(env, 'results', force=True)
env.reset()

print("action space: {0!r}".format(env.action_space))
print("observation space: {0!r}".format(env.observation_space))


def evaluate_nn(net, visualise=False):
    fitness = 0
    observation = env.reset()
    for _ in range(500):
        if visualise:
            env.render()

        n = 8
        pixels = []
        for i in range(int(96 / n)):
            for j in range(int(96 / n)):
                # 4x4 convolution
                px = 0
                for x in range(n):
                    for y in range(n):
                        inside = i * n + y < 210 and j * n + x < 160
                        px += max(observation[i * n + y]
                                  [j * n + x]) if inside else 0

                pixels.append(px / (255 * (n ** 2)))

        steering = net.activate(pixels)
        observation, reward, done, info = env.step(steering)
        fitness += reward
        if done:
            break

    return fitness


def evaluate_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    genome.fitness = evaluate_nn(net, visualise=True)


def evaluate_genomes(genomes, config):
    for gid, genome in genomes:
        evaluate_genome(genome, config)


def run(config_path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-19')

    # setup some output statistics
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every 10 generations or 900 seconds.
    pop.add_reporter(neat.Checkpointer(10, 900))

    # pe = neat.ParallelEvaluator(4, evaluate_genome)
    # winner = pop.run(pe.evaluate, 5)
    winner = pop.run(evaluate_genomes, 150)
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    while True:
        evaluate_nn(net, visualise=True)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-racing')
    run(config_path)

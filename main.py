import matplotlib.pyplot as plt
import numpy as np
import random
import gym
from maze_lagranzh import simulate_mechanical
from maze_q_learning import simulate_qlearning
import solver

random.seed(4)
np.random.seed(4)
NUM_EPISODES = 150
RENDER_MAZE = True


def main():
    env = gym.make("maze-v0")
    env.seed(0)
    maze_size = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    num_actions = env.action_space.n  # ["N", "S", "E", "W"]
    result_history_q = simulate_qlearning(env, maze_size, NUM_EPISODES, RENDER_MAZE)
    plt.plot(result_history_q, label = 'Q-learning')
    maze_solver = solver.MazeSolver(maze_size)
    result_history_l = simulate_mechanical(env, maze_solver, NUM_EPISODES, RENDER_MAZE)
    plt.plot(result_history_l, label = 'Mechanical approach')
    plt.legend()
    plt.grid()
    plt.xlabel('Steps')
    plt.ylabel('Best path')
    plt.show()


if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import gym
from maze_lagranzh import simulate_mechanical
from maze_q_learning import simulate_qlearning
import solver

random.seed(4)
np.random.seed(4)
NUM_EPISODES = 150
RENDER_MAZE = False


def main():
    env = gym.make("maze-v0")
    env.seed(0)
    maze_size = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    num_actions = env.action_space.n  # ["N", "S", "E", "W"]
    result_qlearning = np.zeros(2500)
    result_mechanical = np.zeros(2500)
    for _ in tqdm(range(1000)):
        result_history_q = simulate_qlearning(env, maze_size, NUM_EPISODES, RENDER_MAZE, verbose=False)
        result_qlearning += np.array(result_history_q[:2500]) / 1000
        maze_solver = solver.MazeSolver(maze_size)
        result_history_l = simulate_mechanical(env, maze_solver, NUM_EPISODES, RENDER_MAZE, verbose=False)
        result_mechanical += np.array(result_history_l[:2500]) / 1000
    plt.plot(result_qlearning, label = 'Q-learning')
    plt.plot(result_mechanical, label = 'Mechanical approach')
    plt.legend()
    plt.grid()
    plt.xlabel('Steps')
    plt.ylabel('Best path')
    plt.savefig('C:/Users/AShilov2/Desktop/1000_it.jpg', dpi = 1200)
    plt.show()


if __name__ == '__main__':
    main()

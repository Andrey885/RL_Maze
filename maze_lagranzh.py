import sys
import numpy as np
import math
import pickle
import random
import time
import gym
import gym_maze
import solver

random.seed(4)
np.random.seed(4)
STREAK_TO_END = 100


def simulate_mechanical(env, maze_solver, num_episodes, render_maze, verbose=True):
    # Instantiating the learning related parameters
    decay_factor = np.prod(maze_solver.maze_size, dtype=float) / 10.0
    discount_factor = 0.99
    num_streaks = 0
    max_t = decay_factor * 100
    maze_solver.explore_rate = solver.get_explore_rate(0, decay_factor)
    maze_solver.learning_rate = solver.get_learning_rate(0, decay_factor)

    result_history = []
    best_t = 300
    env.render()
    for episode in range(num_episodes):
        # Reset the environment
        state = env.reset()
        done = False
        total_reward = 0
        t = 0
        while not done:
            # Select an action
            action = maze_solver.select_action(state, env)
            # execute the action
            state, reward, done, _ = env.step(action)
            maze_solver.save_env_reaction(state)
            # Observe the result
            total_reward += reward
            t += 1
            result_history.append(best_t)

            if env.is_game_over():
                sys.exit()
            if render_maze:
                env.render()
            if done:
                maze_solver.save_path_and_reset(t)
                if verbose:
                    print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                          % (episode, t, total_reward, num_streaks))
                if t < best_t:
                    best_t = t
            elif t >= max_t - 1:
                if verbose:
                    print("Episode %d timed out at %d with total reward = %f."
                          % (episode, t, total_reward))

        # It's considered done when it's solved over 120 times consecutively
        if num_streaks > STREAK_TO_END:
            break

        # Update parameters
        maze_solver.explore_rate = solver.get_explore_rate(episode, decay_factor)
        maze_solver.learning_rate = solver.get_learning_rate(episode, decay_factor)
    return result_history


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


def show_simulation(env, maze_solver):
    state = env.reset()
    done = False
    maze_solver.load_state()

    t = 0
    while not done:
        # Select an action
        action = maze_solver.select_action(state, env)
        # execute the action
        state, reward, done, _ = env.step(action)
        maze_solver.save_env_reaction(state)
        # Observe the result
        t += 1
        if env.is_game_over():
            sys.exit()
        env.render()
        time.sleep(0.3)
    print('Final number of iterartions:', t)
    maze_solver.show()

if __name__ == "__main__":
    env = gym.make("maze-v0")
    env.seed(0)
    NUM_EPISODES = 150
    RENDER_MAZE = False
    maze_size = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    num_actions = env.action_space.n  # ["N", "S", "E", "W"]
    maze_solver = solver.MazeSolver(maze_size)
    result_history_l = simulate_mechanical(env, maze_solver, NUM_EPISODES, RENDER_MAZE)
    show_simulation(env, maze_solver)

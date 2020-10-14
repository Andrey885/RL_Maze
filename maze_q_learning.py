import sys
import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
import gym
import gym_maze
import solver

random.seed(4)
np.random.seed(4)


def simulate_qlearning(env, maze_size, num_episodes, render_maze, verbose=True):
    def select_action(state, explore_rate):
        # Select a random action
        if random.random() < explore_rate:
            action = env.action_space.sample()
        # Select the action with the highest q
        else:
            action = int(np.argmax(q_table[state]))
        return action


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
    # Number of discrete states (bucket) per state dimension
    NUM_BUCKETS = maze_size  # one bucket per grid

    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    DECAY_FACTOR = np.prod(maze_size, dtype=float) / 10.0

    MAX_T = np.prod(maze_size, dtype=int) * 100
    STREAK_TO_END = 100
    SOLVED_T = np.prod(maze_size, dtype=int)
    DEBUG_MODE = 0
    ENABLE_RECORDING = False

    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

    '''
    Begin simulation
    '''
    recording_folder = "/tmp/maze_q_learning"

    if ENABLE_RECORDING:
        env.monitor.start(recording_folder, force=True)

    # Instantiating the learning related parameters
    result_history = []
    learning_rate = solver.get_learning_rate(0, DECAY_FACTOR)
    explore_rate = solver.get_explore_rate(0, DECAY_FACTOR)
    discount_factor = 0.99

    num_streaks = 0

    # Render tha maze
    env.render()

    best_t = 300
    for episode in range(num_episodes):

        # Reset the environment
        obv = env.reset()

        # the initial state
        state_0 = state_to_bucket(obv)
        total_reward = 0
        for t in range(MAX_T):

            # Select an action
            action = select_action(state_0, explore_rate)
            # execute the action
            obv, reward, done, _ = env.step(action)

            # Observe the result
            state = state_to_bucket(obv)
            total_reward += reward

            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor * (best_q) - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state

            result_history.append(best_t + 1)  # +1 is because of different step count here
            # Print data
            if render_maze:
                env.render()

            if env.is_game_over():
                sys.exit()

            if done:
                if verbose:
                    print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                          % (episode, t, total_reward, num_streaks))

                if t < best_t:
                    best_t = t
                if t <= SOLVED_T:
                    num_streaks += 1
                else:
                    num_streaks = 0
                break

            elif t >= MAX_T - 1:
                print("Episode %d timed out at %d with total reward = %f."
                      % (episode, t, total_reward))
        # It's considered done when it's solved over 120 times consecutively
        if num_streaks > STREAK_TO_END:
            break

        # Update parameters
        explore_rate = solver.get_explore_rate(episode, DECAY_FACTOR)
        learning_rate = solver.get_learning_rate(episode, DECAY_FACTOR)
    return result_history


if __name__ == "__main__":

    # Initialize the "maze" environment
    # env = gym.make("maze-random-10x10-plus-v0")
    env = gym.make("maze-v0")
    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    simulate_qlearning(env, MAZE_SIZE)

    if ENABLE_RECORDING:
        env.monitor.close()

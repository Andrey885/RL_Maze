import sys
import numpy as np
import math
import random
import time
import gym
import gym_maze
import solver

def simulate(maze_solver):
    # Instantiating the learning related parameters
    discount_factor = 0.99
    num_streaks = 0
    maze_solver.explore_rate = solver.get_explore_rate(0, DECAY_FACTOR)
    maze_solver.learning_rate = solver.get_learning_rate(0, DECAY_FACTOR)

    result_history = []
    best_t = 300
    env.render()
    for episode in range(NUM_EPISODES):
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
            # if episode != 0:
            #     time.sleep(0.2)
            # Print data
            if DEBUG_MODE == 2:
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % maze_solver.explore_rate)
                print("Learning rate: %f" % maze_solver.learning_rate)
                print("Streaks: %d" % num_streaks)
                print("")
            elif DEBUG_MODE == 1:
                if done or t >= MAX_T - 1:
                    print("\nEpisode = %d" % episode)
                    print("t = %d" % t)
                    print("Explore rate: %f" % maze_solver.explore_rate)
                    print("Learning rate: %f" % maze_solver.learning_rate)
                    print("Streaks: %d" % num_streaks)
                    print("Total reward: %f" % total_reward)
                    print("")

            # Render tha maze
            if RENDER_MAZE:
                env.render()

            if env.is_game_over():
                sys.exit()

            if done:
                maze_solver.save_path_and_reset(t)
                print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                      % (episode, t, total_reward, num_streaks))
                if t < best_t:
                    best_t = t
            elif t >= MAX_T - 1:
                print("Episode %d timed out at %d with total reward = %f."
                      % (episode, t, total_reward))

        # It's considered done when it's solved over 120 times consecutively
        if num_streaks > STREAK_TO_END:
            break

        # Update parameters
        maze_solver.explore_rate = solver.get_explore_rate(episode, DECAY_FACTOR)
        maze_solver.learning_rate = solver.get_learning_rate(episode, DECAY_FACTOR)
    np.save('lagranzh5x5.npy', np.array(result_history))

def optimal_action():
    # U - matrix nxn
    # q - list of prev steps coordinates
    # p = - intergral(du/dq * dt)
    p = 0
    return p


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


if __name__ == "__main__":

    # Initialize the "maze" environment
    env = gym.make("maze-v0")

    '''
    Defining the environment related constants
    '''
    # Number of discrete states (bucket) per state dimension
    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_BUCKETS = MAZE_SIZE  # one bucket per grid

    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))


    '''
    Defining the simulation related constants
    '''
    NUM_EPISODES = 150
    MAX_T = np.prod(MAZE_SIZE, dtype=int) * 100
    STREAK_TO_END = 100
    SOLVED_T = np.prod(MAZE_SIZE, dtype=int)
    DEBUG_MODE = 0
    RENDER_MAZE = True
    ENABLE_RECORDING = False
    DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0
    maze_solver = solver.MazeSolver(MAZE_SIZE)
    '''
    Begin simulation
    '''
    recording_folder = "/tmp/maze_q_learning"

    if ENABLE_RECORDING:
        env.monitor.start(recording_folder, force=True)

    simulate(maze_solver)

    if ENABLE_RECORDING:
        env.monitor.close()

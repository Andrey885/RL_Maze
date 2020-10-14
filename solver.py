import numpy as np
import random
import math
import cv2

random.seed(4)
np.random.seed(4)
MIN_EXPLORE_RATE = 0.001
MIN_LEARNING_RATE = 0.2


class MazeSolver:
    def __init__(self, maze_size):
        self.optimal_U = np.ones((maze_size[0] + 2, maze_size[1] + 2))  # the maze solution
        self.road_U = np.ones((maze_size[0] + 2, maze_size[1] + 2))  # penalty to restrict agent from going where it has already been
        self.init_square_borders(maze_size)
        self.optimal_U = self.inf_padding(self.optimal_U)
        self.road_U = self.inf_padding(self.road_U)
        self.encoding = np.array(['N', 'S', 'E', 'W'])
        self.path = []
        self.explore_rate = 0
        self.learning_rate = 0
        self.step_number = 0
        self.maze_size = maze_size
        self.last_expected_position = (0,0)
        self.min_found_step_count = maze_size[0] * maze_size[1]
        self.inference = False

    def save_state(self):
        self.best_U = self.optimal_U

    def load_state(self):
        self.explore_rate = 0
        self.optimal_U = self.best_U
        self.inference = True

    def init_square_borders(self, maze_size):
        self.borders_dict = {}
        for x in range(maze_size[0] + 2):
            for y in range(maze_size[1] + 2):
                self.borders_dict[str((x, y))] = []

    @staticmethod
    def inf_padding(map):
        map[:, 0] = np.inf
        map[:, -1] = np.inf
        map[0, :] = np.inf
        map[-1, :] = np.inf
        return map

    def optimal_step(self, x, y):
        # north, south, east, west
        neighbouring_potentials = np.ones(4) * np.inf
        for i in range(len(neighbouring_potentials)):
            next_position = self.encode_action_to_next_position(i, x, y)
            if not str(next_position) in self.borders_dict[str((x, y))]:
                neighbouring_potentials[i] = self.optimal_U[next_position] + self.road_U[next_position]
        # q'' + dU/dq = 0
        if self.inference:
            print(x, y, neighbouring_potentials, self.optimal_U[x, y], self.optimal_U[x+1, y], self.optimal_U[x, y+1], self.optimal_U[x-1, y], self.optimal_U[x, y-1], self.borders_dict[str((x, y))])
        optimal_steps = np.argwhere(neighbouring_potentials == np.min(neighbouring_potentials))[:, 0]
        optimal_step = np.random.choice(optimal_steps)
        return optimal_step

    def select_action(self, state, env):
        x, y = state
        x = int(x) + 1  # padding
        y = int(y) + 1  # padding
        # self.road_U[x, y] += np.max(self.optimal_U[self.optimal_U != np.inf])
        self.road_U[x, y] += 1#self.step_number
        # self.step_number += 1
        # Select a random action
        if random.random() < self.explore_rate:
            while True:
                action = env.action_space.sample()
                next_position = self.encode_action_to_next_position(action, x, y)
                #  do not try to go into borders
                if self.optimal_U[next_position] != np.inf and str(next_position) not in self.borders_dict[str((x, y))]:
                    break
        # Select the action with the highest q
        else:
            action = self.optimal_step(x, y)
        self.last_expected_position = self.encode_action_to_next_position(action, x, y)
        return self.encoding[action]

    def save_env_reaction(self, new_q):
        x_new, y_new = new_q
        x_new += 1
        y_new += 1
        self.path.append([x_new, y_new])
        if x_new != self.last_expected_position[0] or y_new != self.last_expected_position[1]:
            self.borders_dict[str((x_new, y_new))].append(str(self.last_expected_position))

    def save_path_and_reset(self, step_count):
        self.step_number = 0
        self.road_U *= step_count / self.min_found_step_count
        self.road_U[self.road_U == 1] = np.max(self.road_U[self.road_U != np.inf])
        self.optimal_U *= self.road_U
        self.optimal_U /= np.max(self.optimal_U[self.optimal_U != np.inf])
        self.road_U = np.ones((self.maze_size[0] + 2, self.maze_size[1] + 2))
        self.road_U = self.inf_padding(self.road_U)
        if step_count < self.min_found_step_count:
            self.save_state()
            self.min_found_step_count = step_count

    @staticmethod
    def encode_action_to_next_position(action, x, y):
        if action == 0:
            y -= 1
        elif action == 1:
            y += 1
        elif action == 2:
            x += 1
        elif action == 3:
            x -= 1
        return x, y

def get_explore_rate(t, decay_factor):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/decay_factor)))


def get_learning_rate(t, decay_factor):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/decay_factor)))

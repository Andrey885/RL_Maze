import numpy as np
import random
import math
import cv2

random.seed(4)
MIN_EXPLORE_RATE = 0.001
MIN_LEARNING_RATE = 0.2


class MazeSolver:
    def __init__(self, maze_size):
        self.dUdq = np.zeros((maze_size[0], maze_size[1] + 2))  # +2 is for infinite borders padding
        self.optimal_U = np.ones((maze_size[0] + 2, maze_size[1] + 2))  # the maze solution
        self.road_U = np.ones((maze_size[0] + 2, maze_size[1] + 2))  # penalty to restrict agent from going where it has already been
        self.init_square_borders(maze_size)
        self.dUdq = self.inf_padding(self.dUdq)
        self.optimal_U = self.inf_padding(self.optimal_U)
        self.road_U = self.inf_padding(self.road_U)
        self.encoding = np.array(['N', 'S', 'E', 'W'])
        self.path = []
        self.explore_rate = 0
        self.learning_rate = 0
        self.maze_size = maze_size
        self.last_expected_position = (0,0)
        self.min_found_step_count = np.inf

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
        optimal_steps = np.argwhere(neighbouring_potentials == np.min(neighbouring_potentials))[:, 0]
        optimal_step = np.random.choice(optimal_steps)
        return optimal_step

    def select_action(self, state, env):
        x, y = state
        x = int(x) + 1  # padding
        y = int(y) + 1  # padding
        self.road_U[x, y] += 1
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
        if any(new_q != self.last_expected_position):
            self.borders_dict[str((x_new, y_new))].append(str(self.last_expected_position))
        self.show()

    def show(self):
        map = (self.optimal_U + self.road_U).copy()
        showing_map = np.zeros((map.shape[0], map.shape[1], 3))
        map[map != np.inf] /= np.max(map[map != np.inf])
        map *= 150
        showing_map[(map == np.inf) | (map != map), :] = np.array([0, 0, 150])
        showing_map[:, :, 0] = map
        showing_map = showing_map.astype(np.uint8)
        cv2.imshow('f', cv2.resize(showing_map, (640, 480), interpolation = cv2.INTER_NEAREST))
        cv2.waitKey(1)

    def save_path_and_reset(self, step_count):
        if step_count < self.min_found_step_count:
            self.min_found_step_count = step_count
            self.optimal_U = self.road_U
        # print(np.array(self.path))

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

import itertools
import numpy as np
from shapely.geometry import Point, LineString

class DotBuildingBlocks2D(object):

    def __init__(self, env):
        self.env = env
        # robot field of fiew (FOV) for inspecting points, from [-np.pi/6, np.pi/6]
        self.ee_fov = np.pi / 3

        # visibility distance for the robot's end-effector. Farther than that, the robot won't see any points.
        self.vis_dist = 60.0

    def compute_distance(self, prev_config, next_config): # in A* it will be used to compute the heuristic and the cost .   
        #HW3 2.1
        return np.linalg.norm(np.array(next_config) - np.array(prev_config))

    def sample_random_config(self, goal_prob, goal):
        # HW3 2.1
        # Goal bias
        if np.random.rand() < goal_prob:
            return np.asarray(goal, dtype=float)

        x = np.random.randint(self.env.xlimit[0], self.env.xlimit[1] + 1)
        y = np.random.randint(self.env.ylimit[0], self.env.ylimit[1] + 1)
        return np.array([x, y], dtype=float)

    def config_validity_checker(self, state):
        return self.env.config_validity_checker(state)

    def edge_validity_checker(self, state1, state2):
        return self.env.edge_validity_checker(state1, state2)



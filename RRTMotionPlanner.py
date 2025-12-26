import numpy as np
from RRTTree import RRTTree
import time


class RRTMotionPlanner(object):

    def __init__(self, bb, ext_mode, goal_prob, start, goal):

        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb)
        self.start = start
        self.goal = goal

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        # TODO: HW3 2.2.3
        pass

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        # TODO: HW3 2.2.2
        pass

    def extend(self, near_config, rand_config):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        # HW3 2.2.1
        near = np.asarray(near_config, dtype=float)
        rand = np.asarray(rand_config, dtype=float)

        diff = rand - near
        dist = float(np.linalg.norm(diff))
        if dist == 0.0:
            return None

        if self.ext_mode == "E1":
            return rand

        if self.ext_mode == "E2":
            # Step size (eta). For 2D manipulator joint space: radians.
            # Pick a small eta and mention it in the report.
            eta = 0.2
            step = (eta / dist) * diff
            return near + step
        

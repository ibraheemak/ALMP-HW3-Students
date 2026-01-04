import numpy as np
from RRTTree import RRTTree
import time


class RRTInspectionPlanner(object):

    def __init__(self, bb, start, ext_mode, goal_prob, coverage):

        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb, task="ip")
        self.start = start

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.coverage = coverage

        # set step size - remove for students
        self.step_size = min(self.bb.env.xlimit[-1] / 50, self.bb.env.ylimit[-1] / 200)

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        # TODO: HW3 2.3.3
        pass
        inspected_points = []
        """
        bb.compute_coverage(inspected_points)
        run RRT while 
        """
        # while self.bb.compute_coverage(inspected_points) < self.coverage:
        # sample point
        # extend
        # if added to the tree: add its inspected points to inspected points.
        #  if inspected new points: add this node to list of nodes to visit.
        # compute a path that goes through the nodes to visit until all points are inspected.
        """
        how to get an OK path from this:
        - if a node is a parent of another node in the nodes to visit, make sure all of its children are visited before visiting other branches of the tree.
        - first, bubble the inspected points up the tree. 
            then, when planning a path, choose only children of the tree s.t. all of these nodes are visited. do so recursively.
          after the bubbling, this will be like running an expansion algorithm from the root. 
            at every step we have a list of nodes we need to visit, and every child has a list of nodes it contains.
                we then run this recursively on the children s.t. we explore the full list.
                    we return from the recursion when the list to explore is empty.
            this creates a subtree inside of the original tree. now we just need to traverse the entire tree.
                this will ensure we visit all of the nodes we inspected.  
        """
        """
        for the competition: sample a goal prob with some probability. if successful, continue going for that POI until cant.
            if successful, switch to the next POI and attempt to go for it now.
            if not, pick a random next POI for the next goal prob hit.
        this way we try to branch out as far as we can. if 
        """

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        # TODO: HW3 2.3.1
        pass

    def extend(self, near_config, rand_config):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        # TODO: HW3 2.3.1
        pass


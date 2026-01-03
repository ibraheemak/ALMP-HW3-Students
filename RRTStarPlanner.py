import numpy as np
from RRTTree import RRTTree
from RRTMotionPlanner import RRTMotionPlanner
import time

# class RRTStarPlanner(object):
class RRTStarPlanner(RRTMotionPlanner):

    def __init__(
        self,
        bb,
        ext_mode,
        max_step_size,
        start,
        goal,
        max_itr=None,
        stop_on_goal=None,
        k=None,
        goal_prob=0.01,
        visualizer=None,
    ):
        # Initialize base class (sets bb, tree, start, goal, ext_mode, goal_prob, visualizer)
        super().__init__(bb, ext_mode, goal_prob, start, goal, visualizer)

        # RRT* specific params
        self.max_itr = max_itr
        self.stop_on_goal = stop_on_goal
        self.k = k
        self.max_step_size = max_step_size

    def add_node(self, x_near, x_rand):
        """
        after adding the node:
            check if the path to it can be improved by other ~log(n) neighbors in radius R
            check if it can improve the path to other ~log(n) neighbors in radius R.
        """
        new_config = self.extend(x_near, x_rand)
        parent = x_near # TODO: dif of conf and vertex
        # TODO: use correct distance functions. figure out what is vertex and what is conf and what the difference is
        # Validity checks
        if new_config is None or (
                not self.bb.config_validity_checker(new_config)) or (
                not self.bb.edge_validity_checker(parent, new_config)):
            return None
        # search for better parent
        r_i = 1  # TODO: calculate according to formula from lecture about RRT*, or from other educated guess
        knn = self.tree.get_k_nearest_neighbors(new_config, r_i)
        new_config_cost = self.tree.get_cost_for_config(x_near) + self.bb.compute_distance(x_near, new_config)
        for v in knn:
            better_cost = self.tree.get_cost_for_config(v) + self.bb.compute_distance(new_config, v)
            if better_cost < new_config_cost and self.bb.edge_validity_checker(v, new_config):
                new_config_cost = better_cost
                parent = v
        # add node to tree with best parent
        parent_id = self.tree.get_idx_for_config(parent) # TODO: parent? parent.id? dif between conf and vertex
        new_id = self.tree.add_vertex(new_config)
        edge_cost = self.bb.compute_distance(parent, new_config)
        self.tree.add_edge(parent_id, new_id, edge_cost=edge_cost)
        # make node parent of other nodes (RRT* improvement phase)
        for v in knn:
            edge_cost = self.bb.compute_distance(new_config, v)
            better_cost = new_config_cost + edge_cost
            if better_cost < self.tree.get_cost_for_config(v):
                # improve tree by replacing the edge to v with a new edge.
                self.tree.update_cost_recursive(v, new_config, edge_cost, better_cost)
        return new_id

    def plan(self):
        """
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        """
        # TODO: HW3 3
        have_more_time = True # TODO: figure out how max running time is measured
        iteration = 0
        while have_more_time:
            iteration += 1
            rand_config = self.bb.sample_random_config(self.goal_prob, self.goal)
            near_id, near_config = self.tree.get_nearest_config(rand_config)
            self.add_node(near_config, rand_config)
            # TODO: halt condition. finish func.



    def compute_cost(self, plan):
        # HW3 3
        return RRTMotionPlanner.compute_cost(self, plan)

    def extend(self, x_near, x_rand):
        # HW3 3
        return RRTMotionPlanner.extend(self, x_near, x_rand)







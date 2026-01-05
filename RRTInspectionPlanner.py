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

    # helper: sample random config 
    def sample_config(self):
        # With probability goal_prob, bias sampling near the best-coverage vertex so far
        if np.random.rand() < self.goal_prob and len(self.tree.vertices) > 0:
            best_id = self.tree.max_coverage_id
            base = self.tree.vertices[best_id].config

            # small random perturbation around a good configuration
            noise = np.random.uniform(low=-0.3, high=0.3, size=base.shape)
            q = base + noise

            # wrap angles to [-pi, pi]
            q = (q + np.pi) % (2 * np.pi) - np.pi
            return q

        # Otherwise: uniform random joint angles in [-pi, pi]
        return np.random.uniform(low=-np.pi, high=np.pi, size=self.bb.dim)

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        # HW3 2.3.3
        start_cfg = np.asarray(self.start, dtype=float)
        start_seen = self.bb.get_inspected_points(start_cfg)
        root_id = self.tree.add_vertex(start_cfg, inspected_points=start_seen)
        # If start already satisfies coverage
        if self.bb.compute_coverage(start_seen) >= self.coverage:
            return np.array([start_cfg])


        # ---------- main loop ----------
        while True:

            rand_cfg = self.sample_config()
            near_id, near_cfg = self.tree.get_nearest_config(rand_cfg)
            new_cfg = self.extend(near_cfg, rand_cfg)
            if new_cfg is None:
                continue

            new_cfg = np.asarray(new_cfg, dtype=float)

            # validity checks
            if not self.bb.config_validity_checker(new_cfg):
                continue
            if not self.bb.edge_validity_checker(near_cfg, new_cfg):
                continue

            # update inspected points: parent union current visible
            parent_seen = self.tree.vertices[near_id].inspected_points
            now_seen = self.bb.get_inspected_points(new_cfg)
            total_seen = self.bb.compute_union_of_points(parent_seen, now_seen)

            # add vertex and edge
            new_id = self.tree.add_vertex(new_cfg, inspected_points=total_seen)
            edge_cost = self.bb.compute_distance(np.asarray(near_cfg, float), new_cfg)
            self.tree.add_edge(near_id, new_id, edge_cost=edge_cost)

            # stopping condition: desired coverage reached
            if self.bb.compute_coverage(total_seen) >= self.coverage:
                # backtrack to build path
                path = []
                cur = new_id
                while True:
                    path.append(self.tree.vertices[cur].config)
                    if cur == self.tree.get_root_id():
                        break
                    cur = self.tree.edges[cur]
                path.reverse()
                return np.array(path)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        # HW3 2.3.1
        total_cost = 0
        for i in range(len(plan) - 1):
            total_cost += self.bb.compute_distance(plan[i], plan[i + 1])
        return total_cost

    def extend(self, near_config, rand_config):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        # HW3 2.3.1
        near = np.asarray(near_config, dtype=float)
        rand = np.asarray(rand_config, dtype=float)

        diff = rand - near
        dist = float(np.linalg.norm(diff))
        if dist == 0.0:
            return None

        if self.ext_mode == "E1":
            return rand

        if self.ext_mode == "E2":
            eta = 0.7  # Small step size in radians 
            if dist <= eta:
                return rand 
            step = (eta / dist) * diff
            return near + step

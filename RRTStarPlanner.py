import numpy as np
from RRTTree import RRTTree
from RRTMotionPlanner import RRTMotionPlanner
import time

# TODO: use configs and ids in the right places accordingly.

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

    def update_subtree_cost_recursive(self, parent_id):
        """
        Update costs of all descendants of parent_id (parent cost assumed already correct).
        Works with vertex ids.
        """
        for child_id in self.tree.reverse_edges.get(parent_id, []):
            edge_cost = self.bb.compute_distance(self.tree.vertices[parent_id].config,
                                                 self.tree.vertices[child_id].config)
            self.tree.vertices[child_id].set_cost(self.tree.vertices[parent_id].cost + edge_cost)
            self.update_subtree_cost_recursive(child_id)

    def update_subtree_cost(self, v_id, new_parent_id, edge_cost):
        """
        Reparent vertex v_id to new_parent_id and update costs for v_id subtree.
        """
        old_parent = self.tree.edges[v_id]
        self.tree.remove_edge(old_parent, v_id)
        self.tree.add_edge(new_parent_id, v_id, edge_cost=edge_cost)
        self.update_subtree_cost_recursive(v_id)

    def add_node(self, near_id, x_rand):
        """
        Add a node steered from the vertex with id near_id towards x_rand.
        Returns (new_config, new_id) or None on failure/invalid.
        """
        """
        after adding the node:
            check if the path to it can be improved by other ~log(n) neighbors in radius R
            check if it can improve the path to other ~log(n) neighbors in radius R.
        """
        near_config = self.tree.vertices[near_id].config
        new_config = self.extend(near_config, x_rand)
        parent = near_id

        if not RRTMotionPlanner.validate_checks(self, near_config=near_config, new_config=new_config):
            return None
        # search for better parent
        r_i = 0.7  # took from E2 extend eta. TODO: calculate according to formula from lecture about RRT*, or from other educated guess
        knn = self.tree.get_k_nearest_neighbors(new_config, r_i)
        new_config_cost = self.tree.get_cost_for_config(near_config) + self.bb.compute_distance(x_near, new_config)
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
                self.update_subtree_cost(v, new_config, edge_cost)
        return new_id

    def plan(self):
        """
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        """
        # TODO: HW3 3
        self.tree = RRTTree(self.bb, task="mp")
        root_id = self.tree.add_vertex(np.asarray(self.start, dtype=float))
        start_time = time.time()
        avg_time_secs = 50.0
        max_time_secs = 5.0 * avg_time_secs
        iteration = 0
        goal_id = None
        # run until time budget expires
        while (time.time() - start_time) < max_time_secs:
            iteration += 1
            rand_config = self.bb.sample_random_config(self.goal_prob, self.goal)
            near_id, near_config = self.tree.get_nearest_config(rand_config)
            new_config, new_id = self.add_node(near_config, rand_config)
            if (new_config == np.asarray(self.goal, dtype=float)).all():
                goal_id = new_id
        if goal_id is None:
            self.plan_time = time.time() - start_time
            return np.array([], dtype=float)

        path = self.reconstruct_path(goal_id)
        self.plan_time = time.time() - start_time
        return np.array(path, dtype=float)

    def compute_cost(self, plan):
        # HW3 3
        return RRTMotionPlanner.compute_cost(self, plan)

    def extend(self, x_near, x_rand):
        # HW3 3
        # Use base extend (steering logic kept in RRTMotionPlanner)
        return RRTMotionPlanner.extend(self, x_near, x_rand)


import numpy as np
import heapq


class AStarPlanner(object):
    def __init__(self, bb, start, goal):
        self.bb = bb
        self.start = start
        self.goal = goal

        self.nodes = dict()

        # used for visualizing the expanded nodes
        # make sure that this structure will contain a list of positions (states, numpy arrays) without duplicates
        self.expanded_nodes = []

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''

        # initialize an empty plan.
        plan = []

        # define all directions the agent can take - order doesn't matter here
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (-1, -1), (-1, 1), (1, 1), (1, -1)]

        self.epsilon = 20
        plan = self.a_star(self.start, self.goal)
        return np.array(plan)

    # compute heuristic based on the planning_env
    def compute_heuristic(self, state):
        '''
        Return the heuristic function for the A* algorithm.
        @param state The state (position) of the robot.
        '''
        # HW3 2.1
        return self.bb.compute_distance(state, self.goal)

    def a_star(self, start_loc, goal_loc):
        # HW3 2.1
        start = np.asarray(start_loc, dtype=float)
        goal = np.asarray(goal_loc, dtype=float)

        start_key = (int(start[0]), int(start[1]))
        goal_key  = (int(goal[0]),  int(goal[1]))

        # OPEN priority queue: (f, tie_breaker, state_key)
        open_heap = []
        tie = 0

        # Best-known g for each discovered node (regardless of OPEN/CLOSED)
        g_score = {start_key: 0.0}
        parent = {start_key: None}

        # Track membership explicitly
        closed = set()
        open_set = {start_key}
        open_best_g = {start_key: 0.0}  # used to skip stale heap entries

        # Visualization: unique expanded nodes
        self.expanded_nodes = []
        expanded_set = set()

        # Push start
        f0 = 0.0 + self.epsilon * self.compute_heuristic(np.array(start_key, dtype=float))
        heapq.heappush(open_heap, (f0, tie, start_key))
        tie += 1

        while open_heap:
            _, __, curr_key = heapq.heappop(open_heap)

            # Skip if already processed
            if curr_key in closed:
                continue
            
            # Skip stale heap entries (node was updated with better g after this was pushed)
            if curr_key not in open_set:
                continue
            
            # Skip if this heap entry has outdated g-value
            if open_best_g.get(curr_key, float("inf")) != g_score.get(curr_key, float("inf")):
                continue

            # pop_min: move from OPEN to CLOSED
            open_set.remove(curr_key)
            open_best_g.pop(curr_key, None)
            closed.add(curr_key)

            # Record expansion (unique)
            if curr_key not in expanded_set:
                self.expanded_nodes.append(np.array(curr_key, dtype=float))
                expanded_set.add(curr_key)

            # Goal check
            if curr_key == goal_key:
                break

            curr_state = np.array(curr_key, dtype=float)

            # Expand neighbors (implicit grid graph)
            for dx, dy in self.directions:
                nbr_key = (curr_key[0] + dx, curr_key[1] + dy)
                nbr_state = np.array(nbr_key, dtype=float)

                # State validity (inside bounds and not inside obstacle)
                if not self.bb.config_validity_checker(nbr_state):
                    continue

                # Edge validity (segment does not intersect obstacles)
                if not self.bb.edge_validity_checker(curr_state, nbr_state):
                    continue

                step_cost = self.bb.compute_distance(curr_state, nbr_state)
                g_new = g_score[curr_key] + step_cost

                # Case A: s not in OPEN and not in CLOSED
                if (nbr_key not in open_set) and (nbr_key not in closed):
                    g_score[nbr_key] = g_new
                    parent[nbr_key] = curr_key

                    h = self.compute_heuristic(nbr_state)
                    f = g_new + self.epsilon * h
                    open_set.add(nbr_key)
                    open_best_g[nbr_key] = g_new
                    heapq.heappush(open_heap, (f, tie, nbr_key))
                    tie += 1
                    continue

                # Case B: s in OPEN
                if nbr_key in open_set:
                    if g_new < g_score.get(nbr_key, float("inf")):
                        g_score[nbr_key] = g_new
                        parent[nbr_key] = curr_key

                        h = self.compute_heuristic(nbr_state)
                        f = g_new + self.epsilon * h
                        open_best_g[nbr_key] = g_new
                        heapq.heappush(open_heap, (f, tie, nbr_key))
                        tie += 1
                    continue

                # Case C: s in CLOSED
                # (nbr_key must be in closed here)
                if nbr_key in closed:
                    if g_new < g_score.get(nbr_key, float("inf")):
                        g_score[nbr_key] = g_new
                        parent[nbr_key] = curr_key

                        # re-open: move from CLOSED back to OPEN
                        closed.remove(nbr_key)
                        open_set.add(nbr_key)

                        h = self.compute_heuristic(nbr_state)
                        f = g_new + self.epsilon * h
                        open_best_g[nbr_key] = g_new
                        heapq.heappush(open_heap, (f, tie, nbr_key))
                        tie += 1

        # Reconstruct path
        if goal_key not in parent:
            return []

        path_keys = []
        k = goal_key
        while k is not None:
            path_keys.append(k)
            k = parent[k]
        path_keys.reverse()

        return [np.array(k, dtype=float) for k in path_keys]
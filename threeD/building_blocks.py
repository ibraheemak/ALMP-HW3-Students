import numpy as np


class BuildingBlocks3D(object):
    """
    @param resolution determines the resolution of the local planner(how many intermidiate configurations to check)
    @param p_bias determines the probability of the sample function to return the goal configuration
    """

    def __init__(self, transform, ur_params, env, resolution=0.1):
        self.transform = transform
        self.ur_params = ur_params
        self.env = env
        self.resolution = resolution
        
        self.cost_weights = np.array([0.4, 0.3, 0.2, 0.1, 0.07, 0.05])

        self.single_mechanical_limit = list(self.ur_params.mechanical_limits.values())[-1][-1]

        # pairs of links that can collide during sampling
        self.possible_link_collisions = [
            ["shoulder_link", "forearm_link"],
            ["shoulder_link", "wrist_1_link"],
            ["shoulder_link", "wrist_2_link"],
            ["shoulder_link", "wrist_3_link"],
            ["upper_arm_link", "wrist_1_link"],
            ["upper_arm_link", "wrist_2_link"],
            ["upper_arm_link", "wrist_3_link"],
            ["forearm_link", "wrist_2_link"],
            ["forearm_link", "wrist_3_link"],
        ]

    def sample_random_config(self, goal_prob,  goal_conf) -> np.array:
        """
        sample random configuration
        @param goal_conf - the goal configuration
        :param goal_prob - the probability that goal should be sampled
        """
        # HW2 5.2.1
        uni_sample = np.random.uniform(low=0.0, high=1.0, size=None)
        if uni_sample <= goal_prob:
            return goal_conf
        else:
            # sample a random dist with the same size as the goal conf, limited by the mechanical limit.
            return np.random.uniform(low=-self.single_mechanical_limit, high=self.single_mechanical_limit, size=len(goal_conf))

    def config_validity_checker(self, conf) -> bool:
        """
        Check if a configuration is collision-free.
        Returns False if any collision is detected.
        """
        sphere_coords = self.transform.conf2sphere_coords(conf)
        radii = self.ur_params.sphere_radius

        # ------------------------------------------------------------
        # 1) Plane constraint: x <= 0.4 for all spheres
        # ------------------------------------------------------------
        for spheres in sphere_coords.values():
            if np.any(spheres[:, 0] > 0.4):
                return False

        # ------------------------------------------------------------
        # 2) Self-collision: link-link
        # ------------------------------------------------------------
        for link1, link2 in self.possible_link_collisions:
            s1 = sphere_coords[link1]      # (n1, 3)
            s2 = sphere_coords[link2]      # (n2, 3)
            r_sum = radii[link1] + radii[link2]

            # pairwise distances: broadcasting
            # result shape: (n1, n2)
            dists = np.linalg.norm(s1[:, None, :] - s2[None, :, :], axis=2)

            if np.any(dists < r_sum):
                return False

        # ------------------------------------------------------------
        # 3) Floor + obstacle collisions
        # ------------------------------------------------------------
        obstacles = self.env.obstacles
        obs_r = self.env.radius

        for link in self.ur_params.ur_links:
            spheres = sphere_coords[link]
            r_link = radii[link]

            # floor collision (except shoulder)
            if link != "shoulder_link":
                if np.any(spheres[:, 2] < r_link):
                    return False

            # obstacle collision
            if obstacles is not None and len(obstacles) > 0:
                dists = np.linalg.norm(
                    spheres[:, None, :] - obstacles[None, :, :],
                    axis=2
                )
                if np.any(dists < (r_link + obs_r)):
                    return False

        return True



    def edge_validity_checker(self, prev_conf, current_conf) -> bool:
        '''check for collisions between two configurations - return True if trasition is valid
        @param prev_conf - some configuration
        @param current_conf - current configuration
        '''
        # HW2 5.2.4
        res = min(0.5, self.resolution)
        progress = 0
        # iters = 0
        while True:
            # iters+=1
            conf = prev_conf * (1.0 - progress) + current_conf * progress  # TODO: check that this is the right way to use the resolution
            if not self.config_validity_checker(conf):
                # print("iters = " +str(iters))
                return False
            if progress == 1.0:
                # print("iters = " + str(iters))
                return True
            progress += res
            progress = min(progress, 1.0) # do one last iteration, for the final config.

    def compute_distance(self, conf1, conf2):
        """
        Returns the Edge cost- the cost of transition from configuration 1 to configuration 2
        @param conf1 - configuration 1
        @param conf2 - configuration 2
        """
        return np.dot(self.cost_weights, np.power(conf1 - conf2, 2)) ** 0.5

import numpy as np
import random
import math
import time


# Define the Planner class
class Planner:
    def __init__(self, robot_temp, environment_temp):
        self.robot = robot_temp
        self.environment = environment_temp
        self.target_config = None

    # Function to convert continuous coordinates to cell indices
    def cont_xyz_to_cell(self, x, y, z, cellsize):
        pX = int(x / cellsize)
        pY = int(y / cellsize)
        pZ = int(z / cellsize)
        return pX, pY, pZ

    # Define a function to check if a line segment intersects with obstacles
    def line_intersects_obstacle(self, x1, y1, x2, y2, z1, z2, cellsize):
        x1, y1, z1 = self.cont_xyz_to_cell(x1, y1, z1, cellsize)
        x2, y2, z2 = self.cont_xyz_to_cell(x2, y2, z2, cellsize)
        num_points = max(abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)) + 1
        x_vals = np.linspace(x1, x2, num_points)
        y_vals = np.linspace(y1, y2, num_points)
        z_vals = np.linspace(z1, z2, num_points)

        # Check if any point on the line segment intersects with an obstacle
        for x_cell, y_cell, z_cell in zip(x_vals, y_vals, z_vals):
            for obstacle in self.environment.obstacles:
                if obstacle.collides(
                    np.array([x_cell * cellsize, y_cell * cellsize, z_cell * cellsize])
                ):
                    return True
        return False

    def set_target_config(self, target_config):
        self.target_config = target_config

    def IsValidArmConfiguration(self, theta1, theta2, theta3, cellsize):

        # Get the arm configuration
        (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = self.forward_kinematics(
            theta1, theta2, theta3
        )

        # Check if any part of the arm intersects with obstacles
        isValid = not self.line_intersects_obstacle(
            x1, y1, x2, y2, z1, z2, cellsize
        ) and not self.line_intersects_obstacle(x2, y2, x3, y3, z2, z3, cellsize)

        return isValid

    def forward_kinematics(self, theta1, theta2, theta3):

        # Calculate joint positions
        (x1, y1, z1) = (0, 0, self.robot.joints[0].d)  # Base
        (x2, y2, z2) = (
            x1 + self.robot.joints[1].a * np.cos(theta1) * np.cos(theta2),
            y1 + self.robot.joints[1].a * np.sin(theta1) * np.cos(theta2),
            z1 + self.robot.joints[1].a * np.sin(theta2),
        )  # Third ball
        (x3, y3, z3) = (
            x2 + self.robot.joints[2].a * np.cos(theta1) * np.cos(theta2 + theta3),
            y2 + self.robot.joints[2].a * np.sin(theta1) * np.cos(theta2 + theta3),
            z2 + self.robot.joints[2].a * np.sin(theta2 + theta3),
        )
        return (x1, y1, z1), (x2, y2, z2), (x3, y3, z3)

    class Node:
        def __init__(self, config, parent_index):
            self.config = config
            self.parent_index = parent_index

    def generate_random_configuration(self, numofDOFs, step_size):
        config = np.array([random.uniform(0, 2 * math.pi) for _ in range(numofDOFs)])
        return config

    def compute_distance(self, config1, config2):
        distance = sum((config1[i] - config2[i]) ** 2 for i in range(len(config1)))
        return distance

    def find_nearest_node(self, config, tree, min_dist_limit):
        min_dist = min_dist_limit
        nearest_index = -1
        for i, node in enumerate(tree):
            dist = self.compute_distance(config, node.config)
            if dist < min_dist:
                min_dist = dist
                nearest_index = i
        return nearest_index

    def generate_new_configuration(
        self, random_config, nearest_config, step_size, numofDOFs
    ):
        new_config = np.zeros(numofDOFs)
        for i in range(numofDOFs):
            diff = random_config[i] - nearest_config[i]
            new_config[i] = nearest_config[i] + (diff < 0 and -1 or 1) * (
                abs(diff) < step_size and abs(diff) or step_size
            )
        return new_config

    def inverse_kinematics_2d(self, x, y):
        # Calculate distance from origin to end-effector
        d = math.sqrt(x**2 + y**2)
        L1 = self.robot.joints[1].a
        L2 = self.robot.joints[2].a
        if d > L1 + L2 or d < abs(L1 - L2):
            print("Target position is out of reach")
            return (None, None), (None, None)
        # Calculate angle between the line connecting the end-effector to the origin and the first part of the robot
        # arm (θ1)
        cos_theta1 = (L1**2 + d**2 - L2**2) / (2 * L1 * d)
        sin_theta1 = math.sqrt(1 - cos_theta1**2)
        theta1_down = math.atan2(y, x) - math.atan2(sin_theta1, cos_theta1)
        theta1_up = math.atan2(y, x) - math.atan2(-sin_theta1, cos_theta1)

        # Calculate angle between the second part of the robot arm and the line connecting the end-effector to the
        # origin (θ2)
        cos_theta2 = (L1**2 + L2**2 - d**2) / (2 * L1 * L2)
        theta2_down = math.pi - math.acos(cos_theta2)
        theta2_up = -math.pi + math.acos(cos_theta2)

        return (theta1_down, theta2_down), (theta1_up, theta2_up)

    def rrt_connect_planner(
        self,
        start_config,
        target_pos,
        num_dofs,
        max_iterations,
        step_size,
        direct_num,
        cellsize,
    ):
        """
        target_pos: The target position of the end-effector [x,y,z]
        num_dofs: The number of degrees of freedom of the robot
        max_iterations: The maximum number of iterations for the planner
        step_size: The step size for the planner
        direct_num: parameter for RRT-connect
        """
        theta1 = math.atan2(target_pos[1], target_pos[0])
        x_new = math.sqrt(target_pos[0] ** 2 + target_pos[1] ** 2)
        y_new = target_pos[2] - 1
        (theta2, theta3), (theta4, theta5) = self.inverse_kinematics_2d(x_new, y_new)
        sol_counter = 0
        # if theta2 theta3 failed, use tjeta4 theta5
        while sol_counter < 2:
            if sol_counter == 0:
                target_config = np.array([theta1, theta4, theta5])
                random.seed(time.time())  # Seed the random number generator
                tree_start, tree_target = [], []
                tree_start.append(self.Node(start_config, -1))
                tree_target.append(self.Node(target_config, -1))
                print("Start config:", start_config)
                print("Target config:", target_config)
                min_dist_limit = 10000
                self.set_target_config(target_config)
            else:
                target_config = np.array([theta1, theta2, theta3])
                tree_target = []
                tree_target.append(self.Node(target_config, -1))
                print("Start config:", start_config)
                print("Target config:", target_config)
                min_dist_limit = 10000
                self.set_target_config(target_config)

            for iter in range(max_iterations):
                if random.randint(0, direct_num - 1) == 0:
                    random_config = self.generate_random_configuration(
                        num_dofs, step_size
                    )
                    nearest_start_index = self.find_nearest_node(
                        random_config, tree_start, min_dist_limit
                    )
                    nearest_target_index = self.find_nearest_node(
                        random_config, tree_target, min_dist_limit
                    )
                    new_config_start = self.generate_new_configuration(
                        random_config,
                        tree_start[nearest_start_index].config,
                        step_size,
                        num_dofs,
                    )
                    new_config_target = self.generate_new_configuration(
                        random_config,
                        tree_target[nearest_target_index].config,
                        step_size,
                        num_dofs,
                    )
                else:
                    nearest_target_index = self.find_nearest_node(
                        tree_start[-1].config, tree_target, min_dist_limit
                    )
                    nearest_start_index = self.find_nearest_node(
                        tree_target[-1].config, tree_start, min_dist_limit
                    )
                    new_config_start = self.generate_new_configuration(
                        tree_target[-1].config,
                        tree_start[nearest_start_index].config,
                        step_size,
                        num_dofs,
                    )
                    new_config_target = self.generate_new_configuration(
                        tree_start[-1].config,
                        tree_target[nearest_target_index].config,
                        step_size,
                        num_dofs,
                    )

                if self.IsValidArmConfiguration(
                    new_config_start[0],
                    new_config_start[1],
                    new_config_start[2],
                    cellsize,
                ):
                    tree_start.append(self.Node(new_config_start, nearest_start_index))
                    if self.IsValidArmConfiguration(
                        new_config_target[0],
                        new_config_target[1],
                        new_config_target[2],
                        cellsize,
                    ):
                        tree_target.append(
                            self.Node(new_config_target, nearest_target_index)
                        )

                        # Check if both trees intersect
                        for i, node_start in enumerate(tree_start):
                            for j, node_target in enumerate(tree_target):
                                is_break = 0
                                for k in range(num_dofs):
                                    if (
                                        abs(
                                            node_start.config[k] - node_target.config[k]
                                        )
                                        > step_size
                                    ):
                                        is_break = 1
                                        break
                                if not is_break:
                                    is_reverse = 0
                                    for l in range(num_dofs):
                                        if (
                                            abs(
                                                tree_start[0].config[l]
                                                - start_config[l]
                                            )
                                            > 0.0001
                                        ):
                                            is_reverse = 1
                                            break
                                    if is_reverse:
                                        tree_start, tree_target = (
                                            tree_target,
                                            tree_start,
                                        )
                                        i, j = j, i
                                    start_temp = []
                                    start_temp.append(tree_start[i].config)
                                    start_queue = tree_start[i].parent_index
                                    start_counter = 1
                                    while start_queue != -1:
                                        start_temp.append(
                                            tree_start[start_queue].config
                                        )
                                        start_queue = tree_start[
                                            start_queue
                                        ].parent_index
                                        start_counter += 1

                                    target_temp = []
                                    target_temp.append(tree_target[j].config)
                                    target_queue = tree_target[j].parent_index
                                    target_counter = 1
                                    while target_queue != -1:
                                        target_temp.append(
                                            tree_target[target_queue].config
                                        )
                                        target_queue = tree_target[
                                            target_queue
                                        ].parent_index
                                        target_counter += 1

                                    planlength = start_counter + target_counter
                                    plan = start_temp[::-1] + target_temp
                                    total_cost = sum(
                                        math.sqrt(
                                            sum(
                                                (plan[k][l] - plan[k + 1][l]) ** 2
                                                for l in range(num_dofs)
                                            )
                                        )
                                        for k in range(planlength - 1)
                                    )
                                    print("Total cost of the path:", total_cost)
                                    print(
                                        "Vertices generated:",
                                        len(tree_start) + len(tree_target),
                                    )
                                    # for i in range(planlength):
                                    #     print("Step", i, ":", plan[i])

                                    return plan

                # Swap trees
                tree_start, tree_target = tree_target, tree_start
            sol_counter += 1
        print("No path found")
        return None

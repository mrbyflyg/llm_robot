import numpy as np
import os
import math
import sys
import asyncio
from typing import List, Dict
from robot_environment import (
    Joint,
    RobotArm,
    Environment3D,
    Ball,
    Block,
    Box,
    Object3D,
    wrap_with_plot,
    run_environment_async
)
from llm_utils import load_instruction_file, llm_call
from planner import Planner
from controller import LQR

############
# LLM Init #
############
"""
    GPT-based LLM model Section
        @constant: openai.api_key, the API key for OpenAI
        @constant: descriptor, the motion descriptor file, used to guide the motion plan generation
        @constant: coder, the reward coder file, used to guide the function generation
        @constant: motion_descriptor, descriptor file content
        @constant: reward_coder, coder file content
        @constant: md_temp, motion descriptor temperature
        @constant: rc_temp, reward coder temperature
"""

os.environ["OPENAI_API_KEY"] = "your_api_key"
print("** Successfully OpenAI key authorization **")

descriptor = "./prompt/descriptor_order.txt"
coder = "./prompt/coder_order.txt"

motion_descriptor = load_instruction_file(descriptor)
print("\nWe are using motion descriptor file:\n{}".format(descriptor))

reward_coder = load_instruction_file(coder)
print("\nWe are using coder file:\n{}".format(coder))

md_temp = 0.5  # motion descriptor temperature
rc_temp = 0.2  # reward coder temperature

md_messages = [{"role": "system", "content": motion_descriptor}]
rc_messages = [{"role": "system", "content": reward_coder}]

# first call, to instruct the system what to do
# providing the model with motion or coder generation template
md_response = llm_call(md_messages, temperature=md_temp, model="gpt-3.5-turbo")
rc_response = llm_call(rc_messages, temperature=rc_temp, model="gpt-3.5-turbo")

md_response_mess = md_response.choices[0].message
# print("Message response from Motion Descriptor:\n{}".format(md_response_mess))
rc_response_mess = rc_response.choices[0].message
# print("Message response from Reward Coder:\n{}".format(rc_response_mess))

# add the newly-created response to the previous messages or conversation
# which is an <OpenAIObject at 0x137e44e50> like JSON: {"role": "assistant", "content": "Yes."}
md_messages.append(md_response_mess)
rc_messages.append(rc_response_mess)

#############
#   Init    #
#############
joints = [
    Joint(0, 1, 0, np.pi / 2),  # define a joint with DH parameters (theta, d, a, alpha)
    Joint(0, 0, 1, 0),
    Joint(0, 0, 1, 0, name="joint3"),  # name is optional; useful for debugging
]
robot = RobotArm(joints)
objects = [
    Ball(
        name="apple",
        position=np.array([-0.2, 0, 1.4]),
        radius=0.1,
        is_obstacle=False,
        color=(0.8, 0, 0),
    ),
    Ball(
        name="banana",
        position=np.array([1, 1, 0.5]),
        radius=0.1,
        is_obstacle=False,
        color=(1, 1, 0.5),
    ),
    # Bright red
    Box(
        name="box",
        position=np.array([-0.1, 0.5, 1.2]),
        size=np.array([0.2, 0.2, 0.2]),
        color=(0, 0.5, 0),
    ),
    # Green
    Box(
        name="bowl",
        position=np.array([0.8, 0.8, 1.2]),
        size=np.array([0.2, 0.2, 0.2]),
        color=(0.75, 0.99, 0.99),
    ),
    Ball(
        name="target",
        position=np.array([0.8, 1, 1.4]),
        radius=0.01,
        color=(0, 0.8, 0),
    ),
    # Light green
    Ball(
        name="obstacle_ball1",
        position=np.array([-0.2, 0, 1.3]),
        radius=0.09,
        is_obstacle=True,
        color=(0, 0, 0.8),
    ),
    # Dark blue
    Ball(
        name="obstacle_ball2",
        position=np.array([-0.3, 0, 1.4]),
        radius=0.09,
        is_obstacle=True,
        color=(0.5, 0, 0.5),
    ),
    # Purple
    Block(
        name="obstacle_box2",
        position=np.array([0.5, -0.4, 1.1]),
        size=np.array([0.3, 0.3, 0.3]),
        is_obstacle=True,
        color=(0, 0.5, 0.5),
    ),  # Teal
    Block(
        name="obstacle_box3",
        position=np.array([0.2, 0.5, 1.3]),
        size=np.array([0.15, 0.15, 0.15]),
        is_obstacle=True,
        color=(0.7, 0.3, 0.2),
    ),  # Brown
    Block(
        name="obstacle_box4",
        position=np.array([0.7, 0.3, 1.3]),
        size=np.array([0.15, 0.15, 0.15]),
        is_obstacle=True,
        color=(0, 0, 0.5),
    ),  # Dark blue
]


# struct a class to store current start, target and plan
class Progress:
    def __init__(
        self, start: Object3D, target: Object3D, plan: list | None, planner: Planner
    ):
        self.start = start
        self.target = target
        self.plan = plan
        self.planner = planner
        self.plan_counter = 0
        self.plan_size = len(plan)
        self.follow_end_effector = False
        self.inside_box = None
        self.curr_box_config = None

    def set_follow_end_effector(self, follow_end_effector):
        self.follow_end_effector = follow_end_effector

    def set_plan_counter(self, plan_counter_):
        self.plan_counter = plan_counter_

    def set_inside_box(self, inside_box):
        self.inside_box = inside_box

    def set_curr_box_config(self, curr_box_config):
        self.curr_box_config = curr_box_config


environment = Environment3D(objects, robot)
dt = 0.001
plot_interval = 1 / 24

num_dofs = 3  # Number of degrees of freedom
max_iterations = 3000  # Maximum number of iterations
step_size = 0.01  # Step size
direct_num = 3  # Parameter for RRT-Connect
cellsize = 0.01  # Parameter for RRT-Connect
plan_counter = 0
progress_list: List[Progress] = []
num_progress = 0
box_namelist = ["box", "bowl"]
environment.get_object("box").set_enter_position(np.array([-0.1, 0.5, 1.35]))
environment.get_object("bowl").set_enter_position(np.array([0.8, 0.8, 1.35]))
is_inside_box = False
objects_to_fetch: Dict[int, str] = {}


def update_plan(plan, theta1, theta2, theta3, theta4, theta5, step_size):
    if len(plan) == 0:
        print("Plan is empty.")
        return

    # Extract the last set of angles from the plan
    current_angles = plan[-1]
    # Check the sum condition
    if sum(abs(x - y) for x, y in zip(current_angles, [theta1, theta2, theta3])) > sum(
        abs(x - y) for x, y in zip(current_angles, [theta1, theta4, theta5])
    ):
        # Current angle needs to move towards target_angles
        # Target angles based on your condition
        target_angles = [theta1, theta4, theta5]

    else:
        target_angles = [theta1, theta2, theta3]

    while any(
        abs(current_angle - target_angle) > step_size
        for current_angle, target_angle in zip(current_angles, target_angles)
    ):
        # Generate next set of angles moving towards the target by at most step_size
        next_angles = []
        for current_angle, target_angle in zip(current_angles, target_angles):
            if abs(current_angle - target_angle) <= step_size:
                # If within step_size, just move to target_angle
                next_angle = target_angle
            elif current_angle < target_angle:
                # Increment positively but not beyond target_angle
                next_angle = min(current_angle + step_size, target_angle)
            else:
                # Increment negatively but not beyond target_angle
                next_angle = max(current_angle - step_size, target_angle)
            next_angles.append(next_angle)

        # Append the computed next step to the plan
        plan.append(next_angles)
        # Update current_angles for next iteration
        current_angles = next_angles

    return plan


def set_start2target(name_obj_A, name_obj_B):
    print(
        "Setting reward for minimizing l2_distance between {} and {}".format(
            name_obj_A, name_obj_B
        )
    )

    if name_obj_A != "palm" and name_obj_A not in box_namelist:
        objects_to_fetch[len(progress_list)] = name_obj_A

    planner = Planner(robot, environment)
    if len(progress_list) == 0:
        if name_obj_B in box_namelist:
            print("target is a box")
            plan = planner.rrt_connect_planner(
                np.array(robot.get_params()),
                environment.get_object(name_obj_B).get_enter_position(),
                num_dofs,
                max_iterations,
                step_size,
                direct_num,
                cellsize,
            )
            if plan == None:
                print("Invalid Target Position")
                return
            target_pos = environment.get_object(name_obj_B).get_position()
            theta1 = math.atan2(target_pos[1], target_pos[0])
            x_new = math.sqrt(target_pos[0] ** 2 + target_pos[1] ** 2)
            y_new = target_pos[2] - 1
            (theta2, theta3), (theta4, theta5) = planner.inverse_kinematics_2d(
                x_new, y_new
            )
            if theta2 == None:
                print("Target is out of reach")
                return
            update_plan(plan, theta1, theta2, theta3, theta4, theta5, step_size)
            P = Progress(
                environment.get_object(name_obj_A),
                environment.get_object(name_obj_B),
                plan,
                planner,
            )
            P.set_inside_box(environment.get_object(name_obj_B))
            P.set_curr_box_config(np.array(plan[-1]))
        else:
            plan = planner.rrt_connect_planner(
                np.array(robot.get_params()),
                environment.get_object(name_obj_B).get_position(),
                num_dofs,
                max_iterations,
                step_size,
                direct_num,
                cellsize,
            )
            if plan == None:
                print("Invalid Target Position")
                return
            P = Progress(
                environment.get_object(name_obj_A),
                environment.get_object(name_obj_B),
                plan,
                planner,
            )
        if name_obj_A != "palm":
            P.set_follow_end_effector(True)
        progress_list.append(P)
    else:
        if progress_list[-1].inside_box != None:
            print("start is a box")
            plan_start = []
            plan_start.append(np.array(progress_list[-1].curr_box_config))
            target_pos = progress_list[-1].inside_box.get_enter_position()
            theta1 = math.atan2(target_pos[1], target_pos[0])
            x_new = math.sqrt(target_pos[0] ** 2 + target_pos[1] ** 2)
            y_new = target_pos[2] - 1
            (theta2, theta3), (theta4, theta5) = planner.inverse_kinematics_2d(
                x_new, y_new
            )
            if theta2 == None:
                print("Target is out of reach")
                return
            update_plan(plan_start, theta1, theta2, theta3, theta4, theta5, step_size)
            if name_obj_B in box_namelist:
                print("target is a box")
                plan = planner.rrt_connect_planner(
                    plan_start[-1],
                    environment.get_object(name_obj_B).get_enter_position(),
                    num_dofs,
                    max_iterations,
                    step_size,
                    direct_num,
                    cellsize,
                )
                if plan == None:
                    print("Invalid Target Position")
                    return
                target_pos = environment.get_object(name_obj_B).get_position()
                theta1 = math.atan2(target_pos[1], target_pos[0])
                x_new = math.sqrt(target_pos[0] ** 2 + target_pos[1] ** 2)
                y_new = target_pos[2] - 1
                (theta2, theta3), (theta4, theta5) = planner.inverse_kinematics_2d(
                    x_new, y_new
                )
                update_plan(plan, theta1, theta2, theta3, theta4, theta5, step_size)
                plan = plan_start + plan
                P = Progress(
                    environment.get_object(name_obj_A),
                    environment.get_object(name_obj_B),
                    plan,
                    planner,
                )
                P.set_inside_box(environment.get_object(name_obj_B))
                P.set_curr_box_config(plan[-1])
            else:
                plan = planner.rrt_connect_planner(
                    plan_start[-1],
                    environment.get_object(name_obj_B).get_position(),
                    num_dofs,
                    max_iterations,
                    step_size,
                    direct_num,
                    cellsize,
                )
                if plan == None:
                    print("Invalid Target Position")
                    return
                plan = plan_start + plan
                P = Progress(
                    environment.get_object(name_obj_A),
                    environment.get_object(name_obj_B),
                    plan,
                    planner,
                )
        else:
            plan_start = []
            if name_obj_B in box_namelist:
                print("target is a box")
                plan = planner.rrt_connect_planner(
                    progress_list[-1].planner.target_config,
                    environment.get_object(name_obj_B).get_enter_position(),
                    num_dofs,
                    max_iterations,
                    step_size,
                    direct_num,
                    cellsize,
                )
                if plan == None:
                    print("Invalid Target Position")
                    return
                target_pos = environment.get_object(name_obj_B).get_position()
                theta1 = math.atan2(target_pos[1], target_pos[0])
                x_new = math.sqrt(target_pos[0] ** 2 + target_pos[1] ** 2)
                y_new = target_pos[2] - 1
                (theta2, theta3), (theta4, theta5) = planner.inverse_kinematics_2d(
                    x_new, y_new
                )
                update_plan(plan, theta1, theta2, theta3, theta4, theta5, step_size)
                plan = plan_start + plan
                P = Progress(
                    environment.get_object(name_obj_A),
                    environment.get_object(name_obj_B),
                    plan,
                    planner,
                )
                P.set_inside_box(environment.get_object(name_obj_B))
                P.set_curr_box_config(plan[-1])
            else:
                plan = planner.rrt_connect_planner(
                    progress_list[-1].planner.target_config,
                    environment.get_object(name_obj_B).get_position(),
                    num_dofs,
                    max_iterations,
                    step_size,
                    direct_num,
                    cellsize,
                )
                if plan == None:
                    print("Invalid Target Position")
                    return
                plan = plan_start + plan
                P = Progress(
                    environment.get_object(name_obj_A),
                    environment.get_object(name_obj_B),
                    plan,
                    planner,
                )
        if name_obj_A != "palm":
            P.set_follow_end_effector(True)
        progress_list.append(P)


def set_start2position(name_obj_A, delta_x, delta_y, delta_z):
    print(
        "Setting reward for moving {} by ({}, {}, {})".format(
            name_obj_A, delta_x, delta_y, delta_z
        )
    )
    if name_obj_A != "palm" and name_obj_A not in box_namelist:
        objects_to_fetch[len(progress_list)] = name_obj_A

    target_pos = np.array(environment.get_object(name_obj_A).get_position()) + np.array(
        [delta_x, delta_y, delta_z]
    )
    planner = Planner(robot, environment)
    theta1 = math.atan2(target_pos[1], target_pos[0])
    x_new = math.sqrt(target_pos[0] ** 2 + target_pos[1] ** 2)
    y_new = target_pos[2] - 1
    (theta2, theta3), (theta4, theta5) = planner.inverse_kinematics_2d(x_new, y_new)
    if theta2 is None:
        print("Target is out of reach")
        return
    global cellsize
    if not planner.IsValidArmConfiguration(theta1, theta2, theta3, cellsize):
        if not planner.IsValidArmConfiguration(theta1, theta4, theta5, cellsize):
            print("Invalid Target Position")
            return
    if len(progress_list) == 0:
        plan = planner.rrt_connect_planner(
            np.array(robot.get_params()),
            target_pos,
            num_dofs,
            max_iterations,
            step_size,
            direct_num,
            cellsize,
        )
        P = Progress(environment.get_object(name_obj_A), None, plan, planner)
        if name_obj_A != "palm":
            P.set_follow_end_effector(True)
        progress_list.append(P)
    else:
        if progress_list[-1].inside_box != None:
            print("start is a box")
            plan_start = []
            plan_start.append(np.array(progress_list[-1].curr_box_config))
            target_pos_n = progress_list[-1].inside_box.get_enter_position()
            theta1 = math.atan2(target_pos_n[1], target_pos_n[0])
            x_new = math.sqrt(target_pos_n[0] ** 2 + target_pos_n[1] ** 2)
            y_new = target_pos_n[2] - 1
            (theta2, theta3), (theta4, theta5) = planner.inverse_kinematics_2d(
                x_new, y_new
            )
            if theta2 == None:
                print("Target is out of reach")
                return
            update_plan(plan_start, theta1, theta2, theta3, theta4, theta5, step_size)
            plan = planner.rrt_connect_planner(
                plan_start[-1],
                target_pos,
                num_dofs,
                max_iterations,
                step_size,
                direct_num,
                cellsize,
            )
            if plan == None:
                print("Invalid Target Position")
                return
            plan = plan_start + plan
        else:
            plan = planner.rrt_connect_planner(
                progress_list[-1].planner.target_config,
                target_pos,
                num_dofs,
                max_iterations,
                step_size,
                direct_num,
                cellsize,
            )
        P = Progress(progress_list[-1].target, None, plan, planner)
        if name_obj_A != "palm":
            P.set_follow_end_effector(True)
        progress_list.append(P)


Q_theta = np.eye(3) * 1e6
Q_theta_dot = np.eye(3) * 1e3
zeros = np.zeros((3, 3))
Q = np.concatenate(
    (
        np.concatenate((Q_theta, zeros), axis=1),
        np.concatenate((zeros, Q_theta_dot), axis=1),
    ),
    axis=0,
)
R = np.eye(3) * 1e-3
N = np.zeros((6, 3))
lqr = LQR(Q, R, N, wrap_angle=True)


async def execute_plan():
    def angular_distance_radians(theta1, theta2):
        theta1 = np.array(theta1)
        theta2 = np.array(theta2)
        delta_theta = np.abs((theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi)
        return np.sqrt(np.sum(delta_theta**2))

    def calc_control(
        thetas: List[float], thetas_dot: List[float], progress: Progress
    ) -> np.ndarray:
        target_thetas = np.array(progress.plan[progress.plan_counter + 1])
        thetas = np.array(thetas)
        control = lqr.get_control_target_speed_zero(
            thetas, np.array(thetas_dot), target_thetas, dt
        )

        DISTANCE_THRESHOLD = np.pi / 50
        # print(
        #     "Distance - thresh: ",
        #     angular_distance_radians(thetas, target_thetas) - DISTANCE_THRESHOLD,
        # )
        # print("Plan Counter: ", progress.plan_counter)
        # print("Control: ", control)
        # print(f"Speeds: {thetas_dot[0]:.3f}, {thetas_dot[1]:.3f}, {thetas_dot[2]:.3f}")
        if angular_distance_radians(thetas, target_thetas) < DISTANCE_THRESHOLD:
            progress.set_plan_counter(progress.plan_counter + 1)

        return control

    def policy(env: Environment3D) -> None:
        global num_progress
        if objects_to_fetch.get(num_progress) != None:
            if env.get_robot_arm().fetched_object == None:
                env.get_robot_arm().fetch_object(
                    env.get_object(objects_to_fetch[num_progress])
                )
                objects_to_fetch.pop(num_progress)
        progress = progress_list[num_progress]
        if progress.plan_counter < progress.plan_size - 1:
            target_acceleration = calc_control(
                env.get_robot_arm().get_params(),
                env.get_robot_arm().get_speeds(),
                progress,
            )
            env.get_robot_arm().set_accelerations(target_acceleration.tolist())
        else:
            num_progress += 1

    def policy_raw(env: Environment3D) -> None:
        global num_progress
        if objects_to_fetch.get(num_progress) != None:
            if env.get_robot_arm().fetched_object == None:
                env.get_robot_arm().fetch_object(
                    env.get_object(objects_to_fetch[num_progress])
                )
                objects_to_fetch.pop(num_progress)
        progress = progress_list[num_progress]
        if progress.plan_counter < progress.plan_size - 1:
            target_acceleration = (
                (
                    np.array(progress.plan[progress.plan_counter + 1])
                    - np.array(env.get_robot_arm().get_params())
                )
                / dt
                - np.array(env.get_robot_arm().get_speeds())
            ) / dt
            env.get_robot_arm().set_accelerations(target_acceleration.tolist())
            progress.set_plan_counter(progress.plan_counter + 1)
        else:
            num_progress += 1
            if env.get_robot_arm().fetched_object != None:
                env.get_robot_arm().release_object()

    def is_done(env: Environment3D) -> bool:
        global num_progress

        if num_progress >= len(progress_list):
            if env.get_robot_arm().fetched_object != None:
                env.get_robot_arm().release_object()
            return True

    if len(sys.argv) >= 2 and sys.argv[1] == "--enable-lqr":
        policy_used = policy
    else:
        policy_used = policy_raw

    await run_environment_async(environment, dt, policy_used, 10000, is_done)


#####################################
# Loop for USER and LLM interaction #
#####################################
async def llm_call_async(messages, temperature=0.2, model="gpt-3.5-turbo"):
    return await asyncio.get_event_loop().run_in_executor(
        None, llm_call, messages, temperature, model
    )


async def main_loop():
    while True:
        """
        md message and rc message are appended in turn
        The response from the previous md message is appended
            as the input for the next rc message.
        """
        # Prompt the user for input and store it in a variable
        user_input = await asyncio.get_event_loop().run_in_executor(
            None, input, "User: "
        )
        # add new user command to the Motion Descriptor messages first
        md_messages.append({"role": "system", "content": user_input})

        # Motion descriptor call
        md_response = await llm_call_async(md_messages, temperature=md_temp)
        md_response_mess = md_response.choices[0].message
        md_response_content = md_response_mess.content

        print("\nMotion Descriptor:\n{}".format(md_response_content))
        # From the motion descriptor response message,
        # querying it to the reward coder LLM model call
        rc_messages.append({"role": "system", "content": md_response_content})

        # Reward coder call
        rc_response = await llm_call_async(rc_messages, temperature=rc_temp)
        rc_response_mess = rc_response.choices[0].message
        rc_response_content = rc_response_mess.content

        print("\nCoder:\n{}".format(rc_response_content))
        print("\n\n")

        # add the newly-created response to the previous messages or conversation
        md_messages.append(md_response_mess)
        rc_messages.append(rc_response_mess)

        rc_response_content = rc_response_content.replace("```python", "").strip("` \n")
        code = "{}".format(rc_response_content)
        # Create a local namespace for execution
        local_namespace = {}
        global_namespace = {
            "set_start2target": set_start2target,
            "set_start2position": set_start2position,
            "execute_plan": execute_plan,
        }

        # Execute the code
        async def exec_async():
            exec(
                "async def _async_exec():\n"
                + "".join(f"    {line}\n" for line in code.split("\n")),
                global_namespace,
                local_namespace,
            )
            await local_namespace["_async_exec"]()

        await exec_async()


wrap_with_plot(environment, plot_interval, main_loop)

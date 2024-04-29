from typing import List, Callable, Coroutine
import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import time
import asyncio
from enum import Enum


class Joint:
    def __init__(
        self,
        theta: float = 0,
        d: float = 0,
        a: float = 0,
        alpha: float = 0,
        joint_type: str = "rotational",
        init_speed: float = 0,
        name: str = "",
    ) -> None:
        self.theta: float = theta
        self.d: float = d
        self.a: float = a
        self.alpha: float = alpha
        self.joint_type: str = joint_type
        self.speed: float = init_speed
        self.acceleration: float = 0
        self.transform_matrix: np.ndarray = None
        self.name: str = name
        self._update_transform_matrix()

    def __str__(self) -> str:
        return f"Joint {self.name + ' ' if self.name else self.name}with (theta,d,a,alpha): {self.theta:.2f}, {self.d:.2f}, {self.a:.2f}, {self.alpha:.2f}"

    def set_acceleration(self, acceleration: float) -> None:
        self.acceleration = acceleration

    def update(self, dt: float) -> None:
        if self.joint_type == "rotational":
            self.speed += self.acceleration * dt
            self.theta += self.speed * dt
            self.theta = np.mod(self.theta, 2 * np.pi)
        elif self.joint_type == "prismatic":
            self.d += self.speed * dt
            self.speed += self.acceleration * dt
        self._update_transform_matrix()

    def _update_transform_matrix(self) -> None:
        ct = np.cos(self.theta)
        st = np.sin(self.theta)
        ca = np.cos(self.alpha)
        sa = np.sin(self.alpha)
        self.transform_matrix = np.array(
            [
                [ct, -st * ca, st * sa, self.a * ct],
                [st, ct * ca, -ct * sa, self.a * st],
                [0, sa, ca, self.d],
                [0, 0, 0, 1],
            ]
        )

    def get_speed(self) -> float:
        return self.speed

    def get_param(self) -> float:
        return self.theta if self.joint_type == "rotational" else self.d


class JointConstraint(Enum):
    THETA = "theta"
    D = "d"
    SPEED = "speed"
    ACCELERATION = "acceleration"


class _JointConstraint:
    def __init__(self, kind: JointConstraint, interval: tuple) -> None:
        self.kind: JointConstraint = kind
        self.interval: tuple = interval

    def holds(self, joint: Joint) -> bool:
        if self.kind == JointConstraint.THETA:
            return self.interval[0] <= joint.theta <= self.interval[1]
        elif self.kind == JointConstraint.D:
            return self.interval[0] <= joint.d <= self.interval[1]
        elif self.kind == JointConstraint.SPEED:
            return self.interval[0] <= joint.speed <= self.interval[1]
        elif self.kind == JointConstraint.ACCELERATION:
            return self.interval[0] <= joint.acceleration <= self.interval[1]

    def __str__(self) -> str:
        return f"{self.kind} in interval {self.interval}"

    def print_warning(self, joint) -> None:
        print(f"{joint}: Constraint {self} not satisfied")


class _ConstrainedJoint(Joint):
    def __init__(
        self,
        theta: float = 0,
        d: float = 0,
        a: float = 0,
        alpha: float = 0,
        joint_type: str = "rotational",
        init_speed: float = 0,
        name: str = "",
    ) -> None:
        super().__init__(theta, d, a, alpha, joint_type, init_speed, name)
        self.constraints: List[_JointConstraint] = []

    def update(self, dt: float) -> None:
        super().update(dt)
        for constraint in self.constraints:
            if not constraint.holds(self):
                constraint.print_warning(self)

    def _add_constraint(self, constraint: _JointConstraint) -> None:
        self.constraints.append(constraint)


def get_constrained_joint(
    joint: Joint | _ConstrainedJoint, kind: JointConstraint, interval: tuple
) -> _ConstrainedJoint:
    if isinstance(joint, _ConstrainedJoint):
        joint._add_constraint(_JointConstraint(kind, interval))
    else:
        joint = _ConstrainedJoint(
            joint.theta,
            joint.d,
            joint.a,
            joint.alpha,
            joint.joint_type,
            joint.speed,
            joint.name,
        )
        joint._add_constraint(_JointConstraint(kind, interval))
    return joint


class Plotable:
    def plot(self, window: glfw._GLFWwindow) -> None:
        raise NotImplementedError


def _plot_ball(position: np.ndarray, radius: float, color: str | tuple) -> None:
    glPushMatrix()
    glTranslatef(*position)
    if isinstance(color, str):
        if color == "red":
            glColor3f(1, 0, 0)
        elif color == "green":
            glColor3f(0, 1, 0)
        elif color == "blue":
            glColor3f(0, 0, 1)
        else:
            raise ValueError(f"Invalid color {color}")
    else:
        glColor3f(*color)
    quad = gluNewQuadric()
    gluSphere(quad, radius, 20, 20)
    glPopMatrix()


class Object3D(Plotable):
    def __init__(
        self,
        name: str,
        position: np.ndarray,
        is_obstacle=False,
        color: str | tuple = "red",
        fetchable: bool = True,
    ) -> None:
        self.position: np.ndarray = position
        self.name: str = name
        self.color: str | tuple = color
        self.fetched: bool = False
        self.is_obstacle: bool = is_obstacle
        self.fetchable: bool = fetchable

    def get_name(self) -> str:
        return self.name

    def get_position(self) -> np.ndarray:
        return self.position

    def _follow(self, position: np.ndarray) -> None:
        self.position = position

    def collides(self, position: np.ndarray) -> bool:
        raise NotImplementedError

    def is_fetched(self) -> bool:
        return self.fetched

    def get_enter_position(self) -> np.ndarray:
        return self.enter_position

    def set_enter_position(self, enter_position: np.ndarray) -> None:
        self.enter_position = enter_position

    def plot(self, window: glfw._GLFWwindow) -> None:
        _plot_ball(self.position, 0.1, self.color)


class Ball(Object3D):
    def __init__(
        self,
        name: str,
        position: np.ndarray,
        radius: float,
        is_obstacle=False,
        color: str | tuple = "red",
        fetchable: bool = True,
    ) -> None:
        super().__init__(name, position, is_obstacle, color, fetchable)
        self.radius: float = radius

    def collides(self, position: np.ndarray) -> bool:
        distance = np.linalg.norm(self.position - position)
        return distance <= self.radius

    def plot(self, window: glfw._GLFWwindow) -> None:
        _plot_ball(self.position, self.radius, self.color)


def draw_cube(position: np.ndarray, size: np.ndarray, color: tuple):
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glShadeModel(GL_SMOOTH)
    glPushMatrix()
    glTranslatef(*position)
    glColor3f(*color)
    glBegin(GL_QUADS)

    x, y, z = size[0] / 2, size[1] / 2, size[2] / 2

    glVertex3f(-x, -y, z)
    glVertex3f(x, -y, z)
    glVertex3f(x, y, z)
    glVertex3f(-x, y, z)

    glVertex3f(-x, -y, -z)
    glVertex3f(-x, y, -z)
    glVertex3f(x, y, -z)
    glVertex3f(x, -y, -z)

    glVertex3f(-x, y, -z)
    glVertex3f(-x, y, z)
    glVertex3f(x, y, z)
    glVertex3f(x, y, -z)

    glVertex3f(-x, -y, -z)
    glVertex3f(x, -y, -z)
    glVertex3f(x, -y, z)
    glVertex3f(-x, -y, z)

    glVertex3f(x, -y, -z)
    glVertex3f(x, y, -z)
    glVertex3f(x, y, z)
    glVertex3f(x, -y, z)

    glVertex3f(-x, -y, -z)
    glVertex3f(-x, -y, z)
    glVertex3f(-x, y, z)
    glVertex3f(-x, y, -z)

    glEnd()
    glPopMatrix()

    glDisable(GL_BLEND)


def draw_box(position: np.ndarray, size: np.ndarray, color: tuple):
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glShadeModel(GL_SMOOTH)
    glPushMatrix()
    glTranslatef(*position)
    glColor3f(*color)
    glBegin(GL_QUADS)

    x, y, z = size[0] / 2, size[1] / 2, size[2] / 2

    # glVertex3f(-x, -y, z)
    # glVertex3f(x, -y, z)
    # glVertex3f(x, y, z)
    # glVertex3f(-x, y, z)

    glVertex3f(-x, -y, -z)
    glVertex3f(-x, y, -z)
    glVertex3f(x, y, -z)
    glVertex3f(x, -y, -z)

    glVertex3f(-x, y, -z)
    glVertex3f(-x, y, z)
    glVertex3f(x, y, z)
    glVertex3f(x, y, -z)

    glVertex3f(-x, -y, -z)
    glVertex3f(x, -y, -z)
    glVertex3f(x, -y, z)
    glVertex3f(-x, -y, z)

    glVertex3f(x, -y, -z)
    glVertex3f(x, y, -z)
    glVertex3f(x, y, z)
    glVertex3f(x, -y, z)

    glVertex3f(-x, -y, -z)
    glVertex3f(-x, -y, z)
    glVertex3f(-x, y, z)
    glVertex3f(-x, y, -z)

    glEnd()
    glPopMatrix()

    glDisable(GL_BLEND)


def draw_cube_edges(
    position: np.ndarray, size: np.ndarray, draw_inner_edges: bool = False
):
    glPushMatrix()
    glTranslatef(*position)
    glColor3f(1, 1, 1)
    glLineWidth(2.0)
    x, y, z = size[0] / 2, size[1] / 2, size[2] / 2

    vertices = [
        [-x, -y, -z],
        [x, -y, -z],
        [x, y, -z],
        [-x, y, -z],
        [-x, -y, z],
        [x, -y, z],
        [x, y, z],
        [-x, y, z],
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    if draw_inner_edges:
        squeezed_x = x - 0.01
        squeezed_y = y - 0.01
        vertices += [
            [-squeezed_x, -squeezed_y, -z],
            [squeezed_x, -squeezed_y, -z],
            [squeezed_x, squeezed_y, -z],
            [-squeezed_x, squeezed_y, -z],
            [-squeezed_x, -squeezed_y, z],
            [squeezed_x, -squeezed_y, z],
            [squeezed_x, squeezed_y, z],
            [-squeezed_x, squeezed_y, z],
        ]
        edges += [
            (8, 12),
            (9, 13),
            (10, 14),
            (11, 15),
        ]

    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()
    glPopMatrix()


class Block(Object3D):
    def __init__(
        self,
        name: str,
        position: np.ndarray,
        size: np.ndarray,
        is_obstacle=False,
        color: tuple = (1.0, 1.0, 1.0),
        fetchable: bool = False,
    ):
        super().__init__(name, position, is_obstacle, color, fetchable)
        self.size = np.array(size)

    def collides(self, position: np.ndarray) -> bool:
        min_corner = self.position - self.size / 2
        max_corner = self.position + self.size / 2
        return np.all(position >= min_corner) and np.all(position <= max_corner)

    def plot(self, window: glfw._GLFWwindow) -> None:
        # Set material properties
        ambient = [0.3 * c for c in self.color] + [1.0]
        diffuse = [0.7 * c for c in self.color] + [1.0]
        specular = [1.0, 1.0, 1.0, 1.0]
        shininess = 100.0
        glMaterialfv(GL_FRONT, GL_AMBIENT, ambient)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse)
        glMaterialfv(GL_FRONT, GL_SPECULAR, specular)
        glMaterialf(GL_FRONT, GL_SHININESS, shininess)

        draw_cube(self.position, self.size, self.color)
        draw_cube_edges(self.position, self.size)


class Box(Object3D):
    def __init__(
        self,
        name,
        position,
        size,
        is_obstacle=False,
        color=(1, 1, 1),
        fetchable=False,
    ):
        super().__init__(name, position, is_obstacle, color, fetchable)
        self.size = np.array(size)

    def collides(self, position: np.ndarray) -> bool:
        min_corner = self.position - self.size / 2
        max_corner = self.position + self.size / 2
        return np.all(position >= min_corner) and np.all(position <= max_corner)

    def plot(self, window: glfw._GLFWwindow) -> None:
        # leave the top open
        ambient = [0.3 * c for c in self.color] + [1.0]
        diffuse = [0.7 * c for c in self.color] + [1.0]
        specular = [1.0, 1.0, 1.0, 1.0]
        shininess = 100.0
        glMaterialfv(GL_FRONT, GL_AMBIENT, ambient)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse)
        glMaterialfv(GL_FRONT, GL_SPECULAR, specular)
        glMaterialf(GL_FRONT, GL_SHININESS, shininess)

        draw_cube_edges(self.position, self.size, True)
        draw_box(self.position, self.size, self.color)


class RobotArm(Plotable):
    def __init__(
        self, joints: List[Joint], linewidth: int = 20, joint_radius: float = 0.05
    ) -> None:
        self.joints: List[Joint] = joints
        self.points: np.ndarray = []
        self.linewidth: int = linewidth
        self.joint_radius: float = joint_radius
        self.fetched_object: Object3D | None = None
        self._update_points()

    def set_accelerations(self, accelerations: List[float]) -> None:
        for i, acceleration in enumerate(accelerations):
            self.joints[i].set_acceleration(acceleration)

    def update(self, dt: float) -> None:
        for joint in self.joints:
            joint.update(dt)
        self._update_points()
        eef_position = self.get_end_effector_position()

    # def _update_points(self) -> None:
    #     transform = np.eye(4)
    #     points = [transform[:3, 3]]  # points[0] = [0, 0, 1]
    #     for joint in self.joints:
    #         transform = np.dot(transform, joint.transform_matrix)
    #         points.append(transform[:3, 3])
    #     self.points = np.array(points)
    def _update_points(self) -> None:
        transform = np.eye(4)

        base_translation = np.array([0.0, 0.0, 0.0, 1.0])
        transform[:, 3] += base_translation
        points = [transform[:3, 3]]

        for joint in self.joints:
            transform = np.dot(transform, joint.transform_matrix)
            points.append(transform[:3, 3])

        self.points = np.array(points)

    def get_end_effector_position(self) -> np.ndarray:
        return self.get_joint_position(-1)

    def get_joint_position(self, joint_index: int) -> np.ndarray:
        return self.points[joint_index]

    def get_joint_positions(self) -> List[np.ndarray]:
        return self.points

    def get_param(self, joint_index: int) -> float:
        return self.joints[joint_index].get_param()

    def get_params(self) -> List[float]:
        return [joint.get_param() for joint in self.joints]

    def get_speed(self, joint_index: int) -> float:
        return self.joints[joint_index].get_speed()

    def get_speeds(self) -> List[float]:
        return [joint.get_speed() for joint in self.joints]

    def release_object(self) -> None:
        if self.fetched_object is None:
            print(f"No object fetched to release")
            return
        self.fetched_object.position = self.get_end_effector_position()
        self.fetched_object.fetched = False
        self.fetched_object = None

    def fetch_object(self, object: Object3D) -> None:
        if object.fetched:
            print(f"Trying to fetch object {object.name} that is already fetched")
            return
        end_effector_position = self.get_end_effector_position()
        distance_to_object = np.linalg.norm(object.position - end_effector_position)
        reach_threshold = 0.1
        if distance_to_object <= reach_threshold:
            object.position = end_effector_position
            object.fetched = True
            self.fetched_object = object

    def plot(self, window: glfw._GLFWwindow) -> None:
        glLineWidth(self.linewidth)

        for i in range(len(self.points) - 1):
            glPushMatrix()
            glBegin(GL_LINES)
            glColor3f(1, 1, 1)
            glVertex3fv(self.points[i])
            glVertex3fv(self.points[i + 1])
            glEnd()
            glPopMatrix()

        for point in self.points:
            glPushMatrix()
            _plot_ball(point, self.joint_radius, "green")
            glPopMatrix()


class Environment3D:
    def __init__(self, objects: List[Object3D], robot: RobotArm) -> None:
        self.objects: List[Object3D] = objects
        self.robot: RobotArm = robot
        self.window: glfw._GLFWwindow = _initialize_plot()
        self.obstacles = [obj for obj in self.objects if obj.is_obstacle]
        self.follow_end_effector = None
        self.final_target = None

    def set_follow_end_effector(self, obj: Object3D) -> None:
        self.follow_end_effector = obj

    def update(self, dt: float) -> None:
        self.robot.update(dt)
        for obj in self.objects:
            # if not obj.is_obstacle and obj.fetchable and not obj.fetched:
            #     self.robot.fetch_object(obj)
            if obj.fetched:
                obj._follow(self.robot.get_end_effector_position())

    def plot(self) -> None:
        self.robot.plot(self.window)
        for obj in self.objects:
            obj.plot(self.window)

    def get_robot_arm(self) -> RobotArm:
        return self.robot

    def get_object(self, name) -> Object3D:
        for obj in self.objects:
            if obj.get_name() == name:
                return obj
        return None

    def get_obstacles(self) -> List[Object3D]:
        return self.obstacles


# Global variables
zoom_factor = 1.0
angle_x = 0
angle_y = 0
last_x = 0
last_y = 0
mouse_down = False


def scroll_callback(window, xoffset, yoffset):
    global zoom_factor
    zoom_factor -= yoffset * 0.05
    zoom_factor = max(0.1, min(10, zoom_factor))


def cursor_pos_callback(window, xpos, ypos):
    global last_x, last_y, angle_x, angle_y
    if mouse_down:
        dx = xpos - last_x
        dy = ypos - last_y
        angle_x += dx * 0.5
        angle_y = max(-85, min(85, angle_y + dy * 0.5))
        last_x = xpos
        last_y = ypos


def update_camera():
    global zoom_factor, angle_x, angle_y
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    # Calculate camera position based on spherical coordinates
    r = 5 * zoom_factor
    camX = 0
    camY = -5 * zoom_factor
    camZ = 0.5 * zoom_factor
    gluLookAt(camX, camY, camZ, 0, 0, 0, 0, 0, 1)
    glRotatef(angle_x, 0, 1, 0)
    glRotatef(angle_y, 1, 0, 0)


def draw_floor(floor_size, square_size):
    glBegin(GL_QUADS)
    for x in range(-floor_size, floor_size, square_size):
        for y in range(-floor_size, floor_size, square_size):
            if ((x // square_size) + (y // square_size)) % 2 == 0:
                glColor3f(0.7, 0.75, 0.8)
            else:
                glColor3f(0.3, 0.3, 0.3)

            glVertex3f(x, y, 0)
            glVertex3f(x + square_size, y, 0)
            glVertex3f(x + square_size, y + square_size, 0)
            glVertex3f(x, y + square_size, 0)
    glEnd()


def _initialize_plot() -> glfw._GLFWwindow:
    if not glfw.init():
        raise RuntimeError("Could not initialize GLFW")
    window = glfw.create_window(1600, 1600, "Robot Environment", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create window")

    glfw.make_context_current(window)

    def setup_lighting():
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glMaterialf(GL_FRONT, GL_SHININESS, 50.0)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.5, 0.5, 0.5, 1.0])

    def init_camera():
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, 1, 0.01, 100)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    setup_lighting()
    glEnable(GL_DEPTH_TEST)
    init_camera()
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glClearColor(1.0, 1.0, 1.0, 1.0)  # background color
    return window


async def _plot_async(environment: Environment3D, plot_interval: float) -> asyncio.Task:
    zoom_factor = 1.2
    last_x = 0
    last_y = 0
    angle_x = 0
    angle_y = 0
    mouse_down = False

    def scroll_callback(window, xoffset, yoffset):
        nonlocal zoom_factor
        zoom_factor -= yoffset * 0.2
        zoom_factor = max(0.1, min(10, zoom_factor))

    def mouse_button_callback(window, button, action, mods):
        nonlocal mouse_down, last_x, last_y
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                mouse_down = True
                last_x, last_y = glfw.get_cursor_pos(window)
            elif action == glfw.RELEASE:
                mouse_down = False

    def cursor_pos_callback(window, xpos, ypos):
        nonlocal last_x, last_y, angle_x, angle_y
        if mouse_down:
            dx = xpos - last_x
            dy = ypos - last_y
            angle_x += dx
            angle_y += dy
            last_x = xpos
            last_y = ypos

    glfw.set_scroll_callback(environment.window, scroll_callback)
    glfw.set_mouse_button_callback(environment.window, mouse_button_callback)
    glfw.set_cursor_pos_callback(environment.window, cursor_pos_callback)
    glfw.make_context_current(environment.window)

    def render(environment: Environment3D) -> None:
        update_camera()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(
            2.5 * zoom_factor, -2.5 * zoom_factor, 2.5 * zoom_factor, 1, 0, 1, 0, 0, 1
        )
        # glRotatef(angle_y, 0, 1, 0)
        glRotatef(angle_x, 0, 0, 1)
        draw_floor(floor_size=64, square_size=2)
        environment.plot()
        glfw.swap_buffers(environment.window)
        glfw.poll_events()

    start_time = time.time()
    last_time = start_time
    accumulated_plot_time = 0.0
    while not glfw.window_should_close(environment.window):
        glfw.poll_events()
        current_time = time.time()
        elapsed_time = current_time - last_time
        accumulated_plot_time += elapsed_time

        if accumulated_plot_time >= plot_interval:
            render(environment)
            accumulated_plot_time -= plot_interval

        await asyncio.sleep(0.01)
    glfw.terminate()


async def run_environment_async(
    environment: Environment3D,
    update_interval: float,
    control_function: Callable[[Environment3D], None],
    duration: float,
    finish_function: Callable[[Environment3D], bool] = None,
) -> None:
    start_time = time.time()
    last_time = start_time

    accumulated_update_time = 0.0

    while time.time() - start_time < duration and (
        finish_function is None or not finish_function(environment)
    ):
        current_time = time.time()
        elapsed_time = current_time - last_time

        accumulated_update_time += elapsed_time

        if accumulated_update_time >= update_interval:
            control_function(environment)
            environment.update(update_interval)
            accumulated_update_time -= update_interval

        last_time = current_time

        await asyncio.sleep(0.01)


def wrap_with_plot(
    environment: Environment3D,
    plot_interval: float,
    main_logic_callback: Callable[[], Coroutine],
) -> None:
    async def run():
        plot_task = asyncio.create_task(_plot_async(environment, plot_interval))
        update_task = asyncio.create_task(main_logic_callback())
        await plot_task
        await update_task

    asyncio.run(run())

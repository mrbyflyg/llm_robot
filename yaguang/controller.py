import control
import numpy as np


def _angle_difference(angle1: float, angle2: float) -> float:
    return (angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi


class LQR:
    def __init__(
        self, Q0: np.ndarray, R: np.ndarray, N0: np.ndarray, wrap_angle: bool = False
    ):
        self.set_Q0(Q0)
        self.set_R(R)
        self.set_N0(N0)
        self.wrap_angle = wrap_angle

    def set_Q0(self, Q0: np.ndarray):
        self.Q0 = Q0
        self.Q = np.concatenate(
            (
                np.concatenate((Q0, np.zeros((1, Q0.shape[1]))), axis=0),
                np.zeros((Q0.shape[0] + 1, 1)),
            ),
            axis=1,
        )
        self.Q_theta = Q0[: Q0.shape[0] // 2, : Q0.shape[1] // 2]
        Q_zeros = np.zeros_like(self.Q_theta)

        self.Q_theta = np.concatenate(
            (
                np.concatenate((self.Q_theta, Q_zeros), axis=0),
                np.concatenate((Q_zeros, Q_zeros), axis=0),
            ),
            axis=1,
        )

    def set_R(self, R: np.ndarray):
        self.R = R

    def set_N0(self, N0: np.ndarray):
        self.N0 = N0
        self.N = np.concatenate((N0, np.zeros((1, N0.shape[1]))), axis=0)
        self.N_theta = N0[: N0.shape[0] // 2, :]
        N_zeros = np.zeros_like(self.N_theta)
        self.N_theta = np.concatenate((self.N_theta, N_zeros), axis=0)

    # below are three types of lqr control; however we only get the target speed zero function work
    def get_control(
        self,
        params: np.ndarray,
        params_dot: np.ndarray,
        target_params: np.ndarray,
        dt: float,
        target_params_dot: np.ndarray = None,
    ) -> np.ndarray:
        # TODO
        if target_params_dot is None:
            target_params_dot = np.zeros_like(target_params)
        params = np.array(params)
        params_dot = np.array(params_dot)
        target_params = np.array(target_params)
        target_params_dot = np.array(target_params_dot)
        if (
            params.shape[0] != target_params.shape[0]
            or params_dot.shape[0] != target_params.shape[0]
            or target_params_dot.shape[0] != target_params.shape[0]
        ):
            raise ValueError(
                "params, params_dot, target_params, and target_params_dot must have the same length"
            )

        params = params.reshape(-1, 1)
        target_params = target_params.reshape(-1, 1)
        params_dot = params_dot.reshape(-1, 1)
        target_params_dot = target_params_dot.reshape(-1, 1)

        param_dim = params.shape[0]
        eye_mat = np.eye(param_dim)
        zero_mat = np.zeros((param_dim, param_dim))
        A0 = np.concatenate(
            (
                np.concatenate((eye_mat, zero_mat), axis=0),
                np.concatenate((dt * eye_mat, eye_mat), axis=0),
            ),
            axis=1,
        )
        B0 = np.concatenate((zero_mat, dt * eye_mat), axis=0)
        d = np.concatenate((target_params_dot * dt, np.zeros_like(params_dot)), axis=0)

        A = np.concatenate(
            (
                np.concatenate((A0, np.zeros((1, 2 * param_dim))), axis=0),
                np.concatenate((d, np.array([[1]])), axis=0),
            ),
            axis=1,
        )
        B = np.concatenate((B0, np.zeros((1, param_dim))), axis=0)

        x = np.concatenate((params, params_dot), axis=0)
        x_ref = np.concatenate((target_params, target_params_dot), axis=0)
        z = x - x_ref

        K, _, _ = control.dlqr(A, B, self.Q, self.R, self.N)
        p = np.concatenate((z, np.array([[1]])), axis=0)
        return -np.dot(K, p).flatten()

    def get_control_target_speed_zero(
        self,
        params: np.ndarray,
        params_dot: np.ndarray,
        target_params: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        params = np.array(params)
        params_dot = np.array(params_dot)
        target_params = np.array(target_params)
        if (
            params.shape[0] != target_params.shape[0]
            or params_dot.shape[0] != target_params.shape[0]
        ):
            raise ValueError(
                "params, params_dot, and target_params must have the same length"
            )

        params = params.reshape(-1, 1)
        target_params = target_params.reshape(-1, 1)
        params_dot = params_dot.reshape(-1, 1)

        param_dim = params.shape[0]
        eye_mat = np.eye(param_dim)
        zero_mat = np.zeros((param_dim, param_dim))
        A0 = np.concatenate(
            (
                np.concatenate((eye_mat, zero_mat), axis=0),
                np.concatenate((dt * eye_mat, eye_mat), axis=0),
            ),
            axis=1,
        )
        B0 = np.concatenate((zero_mat, dt * eye_mat), axis=0)

        x = np.concatenate((params, params_dot), axis=0)
        target_params_dot = np.zeros_like(target_params)
        x_ref = np.concatenate((target_params, target_params_dot), axis=0)
        if self.wrap_angle:
            angle_diffs = np.vectorize(_angle_difference)(params, target_params)
            z = np.concatenate((angle_diffs, params_dot - target_params_dot), axis=0)
        else:
            z = x - x_ref

        K0, _, _ = control.dlqr(A0, B0, self.Q0, self.R, self.N0)
        return -np.dot(K0, z).flatten()

    def get_control_target_params_only(
        self,
        params: np.ndarray,
        target_params: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        params = np.array(params)
        target_params = np.array(target_params)
        if params.shape[0] != target_params.shape[0]:
            raise ValueError("params and target_params must have the same length")

        params = params.reshape(-1, 1)
        target_params = target_params.reshape(-1, 1)

        param_dim = params.shape[0]
        eye_mat = np.eye(param_dim)
        zero_mat = np.zeros((param_dim, param_dim))
        A0 = np.concatenate(
            (
                np.concatenate((eye_mat, zero_mat), axis=0),
                np.concatenate((dt * eye_mat, eye_mat), axis=0),
            ),
            axis=1,
        )
        B0 = np.concatenate((zero_mat, dt * eye_mat), axis=0)

        params_dot = np.zeros_like(params)
        x = np.concatenate((params, params_dot), axis=0)
        target_params_dot = np.zeros_like(target_params)
        x_ref = np.concatenate((target_params, target_params_dot), axis=0)
        if self.wrap_angle:
            angle_diffs = np.vectorize(_angle_difference)(params, target_params)
            z = np.concatenate((angle_diffs, params_dot - target_params_dot), axis=0)
        else:
            z = x - x_ref

        K_theta, _, _ = control.dlqr(A0, B0, self.Q_theta, self.R, self.N_theta)
        return -np.dot(K_theta, z).flatten()

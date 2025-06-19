import numpy as np

class KalmanFilter:
    """Simple constant velocity Kalman filter."""

    def __init__(self, dt: float = 1.0, process_var: float = 1e-2, measurement_var: float = 1.0) -> None:
        self.dt = dt
        # State: [x, y, vx, vy]
        self.x = np.zeros(4)
        self.P = np.eye(4)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        self.Q = np.eye(4) * process_var
        self.R = np.eye(2) * measurement_var

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]

    def update(self, z: np.ndarray) -> None:
        z = np.asarray(z)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

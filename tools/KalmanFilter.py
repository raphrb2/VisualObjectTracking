import numpy as np

# Kalman Filter Class for TP1

class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        self.u = np.array([[u_x], [u_y]])
        self.x = np.zeros((4, 1))
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.B = np.array([[(dt ** 2) / 2, 0],
                           [0, (dt ** 2) / 2],
                           [dt, 0],
                           [0, dt]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = np.array([[(dt ** 4) / 4, 0, (dt ** 3) / 2, 0],
                           [0, (dt ** 4) / 4, 0, (dt ** 3) / 2],
                           [(dt ** 3) / 2, 0, dt ** 2, 0],
                           [0, (dt ** 3) / 2, 0, dt ** 2]]) * std_acc ** 2
        self.R = np.array([[x_std_meas ** 2, 0],
                           [0, y_std_meas ** 2]])
        self.P = np.eye(4)

    def predict(self):
        self.x = self.A @ self.x + self.B @ self.u
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        if z.shape != (2, 1):
            z = np.reshape(z, (2, 1))

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.x.flatten()
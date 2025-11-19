# smoother.py

import numpy as np

class EMASmoother:
    """
    对二维（x, y）坐标做指数移动平均 (EMA) 平滑
    Usage: 每帧调用 .smooth((x, y)) 返回平滑后的 (x, y)
    """
    def __init__(self, alpha=0.7):
        # alpha 越大，当前值影响越大（响应快），alpha 越小，更平滑但延迟大
        self.alpha = alpha
        self.prev = None

    def smooth(self, point):
        # point: tuple or list (x, y)
        x, y = point
        if self.prev is None:
            self.prev = np.array([x, y], dtype=float)
            return (x, y)
        self.prev = self.alpha * np.array([x, y]) + (1 - self.alpha) * self.prev
        return (float(self.prev[0]), float(self.prev[1]))


class KalmanSmoother:
    """
    使用简单二维线性卡尔曼滤波 (constant velocity model) 平滑 x,y 坐标
    状态维度：x, y, vx, vy
    """
    def __init__(self, dt=1.0, process_variance=1e-2, measurement_variance=1e-1):
        # dt: 时间步长 (帧间隔)
        self.dt = dt
        # 状态转移矩阵
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])
        # 观测矩阵 (只测量 x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        # 过程噪声协方差 Q
        self.Q = (process_variance * np.eye(4))
        # 测量噪声协方差 R
        self.R = (measurement_variance * np.eye(2))
        # 初始状态协方差矩阵 P
        self.P = np.eye(4)
        # 初始状态 (x, y, vx, vy)
        self.x = np.zeros((4,))

    def smooth(self, point):
        # 预测
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q

        # 更新
        z = np.array([point[0], point[1]])
        y = z - self.H.dot(self.x)  # 误差
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))  # 卡尔曼增益
        self.x = self.x + K.dot(y)
        I = np.eye(self.F.shape[0])
        self.P = (I - K.dot(self.H)).dot(self.P)

        smoothed_x = float(self.x[0])
        smoothed_y = float(self.x[1])
        return (smoothed_x, smoothed_y)

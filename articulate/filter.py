r"""
    Temporal filters.
"""


__all__ = ['LowPassFilter', 'LowPassFilterRotation', 'KalmanFilter']


from . import math as M
import quaternion     # package: numpy-quaternion
import torch


class KalmanFilter:
    r"""
    https://zhuanlan.zhihu.com/p/113685503
    """
    def __init__(self, F, H, B, Q=None, R=None, x0=None, P=None):
        r"""
        x <- Fx + Bu + N(0, Q)
        y = Hx + N(0, R)

        state x in shape [n]; observation y in shape [m]; input u in shape [k]

        :param F: Tensor in shape [n, n].
        :param H: Tensor in shape [m, n].
        :param B: Tensor in shape [n, k].
        :param Q: Tensor in shape [n, n].
        :param R: Tensor in shape [m, m].
        :param x0: Tensor in shape [n]. Initial state x0.
        :param P: Tensor in shape [n, n]. Covariance matrix of the initial state x0's noise.
        """
        self.n = F.shape[0]
        self.m = H.shape[0]
        self.k = B.shape[1]

        self.F = F
        self.H = H
        self.B = B
        self.Q = torch.eye(self.n) if Q is None else Q
        self.R = torch.eye(self.m) if R is None else R
        self.P = torch.eye(self.n) if P is None else P
        self.x = torch.zeros(self.n, 1) if x0 is None else x0.view(self.n, 1)

    def reset(self, x0=None, P=None):
        r"""
        Reset the state.

        :param x0: Tensor in shape [n]. Initial state x0.
        :param P: Tensor in shape [n, n]. Covariance matrix of the initial state x0's noise.
        """
        self.P = torch.eye(self.n) if P is None else P
        self.x = torch.zeros(self.n, 1) if x0 is None else x0.view(self.n, 1)

    def predict(self, u, Q=None):
        r"""
        Predict the next state.

        :param u: Input u in shape [k].
        :param Q: State noise covariance matrix in shape [n, n]. If None, use the init value.
        :return: Predicted state in shape [n].
        """
        Q = Q or self.Q
        self.x = self.F.mm(self.x) + self.B.mm(u.view(self.k, 1))
        self.P = self.F.mm(self.P).mm(self.F.t()) + Q
        return self.x.clone().view(self.n)

    def correct(self, y, R=None):
        r"""
        Correct the state estimate.

        :param y: Observation in shape [m].
        :param R: Observation noise covariance matrix in shape [m, m]. If None, use the init value.
        :return: Corrected state in shape [n].
        """
        R = R or self.R
        S = R + self.H.mm(self.P.mm(self.H.t()))
        K = self.P.mm(self.H.t()).mm(S.inverse())
        self.x = self.x + K.mm(y.view(self.m, 1) - self.H.mm(self.x))
        self.P = (torch.eye(self.n) - K.mm(self.H)).mm(self.P)
        return self.x.clone().view(self.n)


class LowPassFilter:
    r"""
    Low-pass filter by exponential smoothing.
    """
    def __init__(self, a=0.8):
        r"""
        Current = Lerp(Last, Current, a)

        :math:`y_t = ax_t + (1 - a)y_{t-1}, a \in [0, 1]`
        """
        self.a = a
        self.x = None

    def __call__(self, x):
        r"""
        Smooth the current value x.
        """
        if self.x is None:
            self.x = x
        else:
            self.x = M.lerp(self.x, x, self.a)
        return self.x

    def reset(self):
        r"""
        Reset the filter states.
        """
        self.x = None


class LowPassFilterRotation(LowPassFilter):
    r"""
    Low-pass filter for rotations by exponential smoothing.
    """
    def __init__(self, a=0.8):
        r"""
        Current = Lerp(Last, Current, a)
        """
        super().__init__(a)

    def __call__(self, x):
        r"""
        Smooth the current rotations x.

        :param x: Tensor that can reshape to [n, 3, 3] for rotation matrices.
        """
        qs = quaternion.from_rotation_matrix(x.detach().cpu().numpy(), nonorthogonal=True).ravel()
        if self.x is None:
            self.x = qs
        else:
            for i in range(len(qs)):
                self.x[i] = quaternion.np.slerp_vectorized(self.x[i], qs[i], self.a)
        x = torch.from_numpy(quaternion.as_rotation_matrix(self.x)).float().view_as(x)
        return x

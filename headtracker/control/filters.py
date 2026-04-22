"""Signal processing helpers for head-angle smoothing and cursor mapping."""
import numpy as np


class OneEuroFilter:
    """Adaptive low-pass filter: smooth when still, responsive when moving."""

    def __init__(self, min_cutoff=0.15, beta=0.15, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            return x
        dt = max(1e-3, t - self.t_prev)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


def precision_curve(norm, knee=0.45, knee_out=0.3):
    """Piecewise-linear map: fast in the center (navigation), slow at edges (precision)."""
    s = np.sign(norm)
    a = abs(norm)
    if a <= knee:
        out = (a / knee) * knee_out
    else:
        out = knee_out + ((a - knee) / (1.0 - knee)) * (1.0 - knee_out)
    return s * out


def soft_deadzone(val, inner_dz, outer_dz):
    """Zero inside inner_dz; quadratic ramp to outer_dz. Eliminates binary snap."""
    s = np.sign(val)
    a = abs(val)
    if a <= inner_dz:
        return 0.0
    if a >= outer_dz:
        return s * (a - inner_dz)
    t = (a - inner_dz) / (outer_dz - inner_dz)
    return s * (a - inner_dz) * (t * t)

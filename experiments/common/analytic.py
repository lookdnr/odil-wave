import numpy as np
import scipy.integrate as si


def ricker(t: np.ndarray, f0: float, t0: float) -> np.ndarray:
    """Compute the Ricker wavelet in time"""
    arg = (np.pi * f0 * (t - t0)) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def _greens_2d(t: float, tau: float, r: float, c: float) -> float:
    """Compute the value of the Green's function for the 2D wave equation"""
    s = t - tau
    a = r / c
    if s <= a:
        return 0.0
    return 1.0 / (2.0 * np.pi * np.sqrt(s**2 - a**2))  # heaviside is 1 for t > 0


def _convolve_greens_ricker(
    t: float, a: float, f0: float, t0: float, r: float, c: float, limit: int = 200
) -> float:
    """Compute the integral representing the convolution of the
    Ricker wavelet with the Green's function"""

    def integrand(tau):
        return ricker(tau, f0, t0) * _greens_2d(t, tau, r, c)

    val, _ = si.quad(
        integrand, 0.0, t - a, limit=limit
    )  # integrate between 0 and t - a
    return val


def analytical_u_single_time(
    t: float, r: float, c: float, f0: float, t0: float, limit=200
) -> float:
    """Compute the analytical wavefield at time t.
    Recovers u by means of convolving the Ricker wavelet with the Green's function.
    """
    a = r / c
    if t <= a:
        return 0.0

    u = _convolve_greens_ricker(t, a, f0, t0, r, c, limit)
    return u


def analytical_trace(
    times: np.ndarray, r: float, c: float, f0: float, t0: float
) -> np.ndarray:
    return np.array([analytical_u_single_time(t, r, c, f0, t0) for t in times])

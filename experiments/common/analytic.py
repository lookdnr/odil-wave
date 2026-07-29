import numpy as np
import scipy.integrate as si
from .config import RunConfig


def ricker(t: np.ndarray, f0: float, t0: float) -> np.ndarray:
    """Compute the Ricker wavelet in time"""
    arg = (np.pi * f0 * (t - t0)) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def src_rec_distance(cfg: RunConfig) -> np.ndarray:
    """Compute the src-rec distance r = sqrt((x_rec - x_src)^2 + (y_rec - y_src)^2)
    for one source location and an (n_recv, 2) array of receiver locations"""
    src_coords = np.asarray(cfg.source_loc)
    rec_coords = np.asarray(cfg.recv_locs)
    r2 = np.sum((rec_coords - src_coords) ** 2, axis=-1)
    return np.sqrt(r2)


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


def analytical_traces(cfg: RunConfig, times: np.ndarray, c: float) -> np.ndarray:
    """Compute the analytical traces at multiple receivers for a single src.

    Returns an (nt, n_recv) array
    """
    distances = np.atleast_1d(src_rec_distance(cfg))
    traces = [
        [analytical_u_single_time(t, r, c, cfg.f0, cfg.t0) for t in times]
        for r in distances
    ]
    return np.array(traces).T

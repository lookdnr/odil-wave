import numpy as np
import numpy.polynomial.legendre as npl
from odil_wave import Sources, Receivers

# compute the sample points and weights for Gauss-Legendre quadrature
POINTS, WEIGHTS = npl.leggauss(256)


def ricker(t: np.ndarray, f0: float, t0: float) -> np.ndarray:
    """Compute the Ricker wavelet in time"""
    arg = (np.pi * f0 * (t - t0)) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def src_rec_distance(src: Sources, recvs: Receivers) -> np.ndarray:
    """Compute the src-rec distance r = sqrt((x_rec - x_src)^2 + (y_rec - y_src)^2)
    for one source location and an (n_recv, 2) array of receiver locations"""
    src_coords = np.asarray(src.src_xy)
    rec_coords = np.asarray(recvs.recv_xy)
    r2 = np.sum((rec_coords - src_coords) ** 2, axis=-1)
    return np.sqrt(r2)


def _gl_integrate(f, lo: float, hi: float) -> float:
    """Compute the Gauss-Legendre integral of f over [lo, hi] interval

    Follows after:

    https://runebook.dev/en/docs/numpy/reference/generated/numpy.polynomial.legendre.leggauss
    """
    # leggaus roots and weights are for integrating over the standard interval
    # [-1, 1], so we must scale and shift the points and weights
    points = 0.5 * (hi - lo) * POINTS + 0.5 * (hi + lo)
    weights = 0.5 * (hi - lo) * WEIGHTS
    return np.sum(f(points) * weights)


def analytical_trace(
    r: float, times: np.ndarray, c: float, f0: float, t0: float
) -> np.ndarray:
    """Exact 2D point source trace at radius r via g conv ricker (cosh substituted)"""
    u = np.zeros_like(times)

    # arrival time: time taken for wave with speed c to move distance r
    a = r / c

    # arrival filter
    live = times > a
    if not live.any():
        return u

    # integrate over time for limits [0, acosh(t/a)]
    for i, t in enumerate(times):
        if t > a:
            u[i] = _gl_integrate(
                lambda nu: ricker(t - a * np.cosh(nu), f0, t0),
                lo=0.0,
                hi=np.arccosh(t / a),
            )

    return u / (2 * np.pi)  # scale according to integral


def analytical_traces(
    src: Sources, recvs: Receivers, times: np.ndarray, c: float
) -> np.ndarray:
    """Compute the analytical traces at multiple receivers for a single src.

    Returns an (nt, n_recv)

    Uses Gauss-Legendre quadrature over the integral int s * g dtau,
    where g is in its cosh-substituted form to avoid the 1/sqrt(s^2 - a^2) singularity
    """
    distances = src_rec_distance(src, recvs)
    traces = [analytical_trace(r, times, c, src.f0, src.t0) for r in distances]
    return np.array(traces).T

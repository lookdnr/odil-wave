import numpy as np
import math
from typing import Tuple
import scipy.special as sps

from odil_wave import Grid


def place_ellipse(
    grid: Grid,
    n_locations: int,
    ring_centre: Tuple[float, float] = (0.0, 0.0),
    a_frac: float = 0.5,
    b_frac: float = 0.5,
) -> np.ndarray:
    """Compute grid indices lying on an ellipse within the domain.

    Parameters
    ----------
    grid : Grid
        Grid the ellipse is placed on.
    n_locations : int
        Number of points to distribute uniformly around the ellipse.
    ring_centre : tuple of (float, float), optional
        Centre of the ellipse in spatial coordinates.
    a_frac, b_frac : float, optional
        Semi axis fractions of the ellipse.

    Returns
    -------
    np.ndarray
        (n_locations, 2) array of nearest node grid indices.
    """
    (xmin, xmax), (ymin, ymax) = grid.extent
    cx, cy = ring_centre
    a = a_frac * (xmax - xmin) / 2.0
    b = b_frac * (ymax - ymin) / 2.0

    k = np.arange(n_locations)
    theta = 2.0 * math.pi * k / n_locations
    x_k = cx + a * np.cos(theta)
    y_k = cy + b * np.sin(theta)

    i = np.round((x_k - xmin) / grid.dx).clip(0, grid.nx - 1).astype(int)
    j = np.round((y_k - ymin) / grid.dy).clip(0, grid.ny - 1).astype(int)

    return np.stack([i, j], axis=-1)


# Kaiser window parameters, half width r : beta
# matches devito table
_HICKS_BETA = {
    2: 2.94,
    3: 4.53,
    4: 4.14,
    5: 5.26,
    6: 6.40,
    7: 7.51,
    8: 8.56,
    9: 9.56,
    10: 10.64,
}


def _kaiser(offset: np.ndarray, R: float, beta: float):
    """Kaiser window function that modulates the sinc function

    offset = x - x_n, where x is the arbitrary src position and
    x_n is the position of the nth grid point
    R is the half width of the window and beta is the Kaiser window
    shape parameter

    see https://doi.org/10.1190/1.1451454
    """
    w = np.zeros_like(offset, dtype=float)
    m = np.abs(offset) <= R

    # I0 is a modified Bessel function
    w[m] = sps.i0(beta * np.sqrt(1.0 - (offset[m] / R) ** 2)) / sps.i0(beta)
    return w


def _sinc_weights(grid, x_s: float, y_s: float, n_sinc: int) -> np.ndarray:
    """(nx, ny) sinc interpolation weights for a point source at (x_s, y_s),
    modulated by the Kaiser window function.

    Each weight is
        sinc((x_s - x[i])/dx) * sinc((y_s - y[j])/dy) * kaiser(offset, r, beta),
    computed over a window of n_sinc nodes per dimension centred on
    the nearest node.
    """
    (xmin, _), (ymin, _) = grid.extent

    # grid indices of the source position
    fi = (x_s - xmin) / grid.dx
    fj = (y_s - ymin) / grid.dy

    # window of node indices centred on nearest node
    half = n_sinc // 2
    beta = _HICKS_BETA[half]  # Kaiser shape parameter for this half width

    # centre the window on the source
    i_win = np.arange(int(np.ceil(fi - half)), int(np.floor(fi + half)) + 1)
    j_win = np.arange(int(np.ceil(fj - half)), int(np.floor(fj + half)) + 1)

    # mask out indices that fall outside the grid
    i_mask = (i_win >= 0) & (i_win < grid.nx)
    j_mask = (j_win >= 0) & (j_win < grid.ny)

    # 1-D Kaiser-windowed sinc weights
    d_i, d_j = fi - i_win, fj - j_win
    wi = np.sinc(fi - i_win) * _kaiser(d_i, half, beta)
    wj = np.sinc(fj - j_win) * _kaiser(d_j, half, beta)

    patch = np.outer(wi[i_mask], wj[j_mask])  # weights (before write)

    # normalize so the discrete integral sum(w*dx*dy) = 1 (unit Dirac delta),
    # correcting for sinc tail truncation and boundary clipping

    patch /= patch.sum() * grid.dx * grid.dy  # weights (normalised)

    # crete weights matrix
    W = np.zeros((grid.nx, grid.ny))
    W[np.ix_(i_win[i_mask], j_win[j_mask])] = patch
    return W


def build_weight_matrix(grid, xy, n_objects, n_sinc: int) -> np.ndarray:
    """Precompute the sinc injection/ extraction weight matrix for a set of points.

    Built so that multiplying by `W` encodes injection and by `W.T` encodes
    extraction, giving adjoint safety for the inverse problem.

    Parameters
    ----------
    grid : Grid
        Grid the points live on.
    xy : np.ndarray
        (n_objects, 2) array of spatial (x, y) coordinates.
    n_objects : int
        Number of points (sources or receivers).
    n_sinc : int
        Width of the sinc interpolation window per dimension.

    Returns
    -------
    np.ndarray
        (nx*ny, n_objects) weight matrix.
    """
    W = np.zeros((grid.nx * grid.ny, n_objects))

    for s in range(n_objects):
        x_s, y_s = xy[s]
        W[:, s] = _sinc_weights(grid, x_s, y_s, n_sinc).ravel()
    return W

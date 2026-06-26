import numpy as np
import math
from typing import Tuple

from odil_wave import Grid


def place_ellipse(
    grid: Grid,
    n_locations: int,
    ring_centre: Tuple[float, float] = (0.0, 0.0),
    a_frac: float = 0.5,
    b_frac: float = 0.5,
) -> np.ndarray:
    """Return (n, 2) integer full-grid indices on an ellipse inside the interior."""
    (ix_min, ix_max), (iy_min, iy_max) = grid.interior_extent
    cx, cy = ring_centre
    a = a_frac * (ix_max - ix_min) / 2.0
    b = b_frac * (iy_max - iy_min) / 2.0

    k = np.arange(n_locations)
    theta = 2.0 * math.pi * k / n_locations
    x_k = cx + a * np.cos(theta)
    y_k = cy + b * np.sin(theta)

    (xmin, _), (ymin, _) = grid.extent
    i = np.round((x_k - xmin) / grid.dx).clip(0, grid.nx - 1).astype(int)
    j = np.round((y_k - ymin) / grid.dy).clip(0, grid.ny - 1).astype(int)

    return np.stack([i, j], axis=-1)


def _sinc_weights(grid, x_s: float, y_s: float, n_sinc: int) -> np.ndarray:
    """(nx, ny) sinc interpolation weights for a point source at (x_s, y_s).

    Each weight is sinc((x_s - x[i])/dx) * sinc((y_s - y[j])/dy), computed
    over a window of n_sinc nodes per dimension centred on the nearest node.
    """
    (xmin, _), (ymin, _) = grid.extent

    # grid indices of the source position
    fi = (x_s - xmin) / grid.dx
    fj = (y_s - ymin) / grid.dy

    # window of node indices centred on nearest node
    half = n_sinc // 2
    i_win = np.arange(int(round(fi)) - half, int(round(fi)) + half)
    j_win = np.arange(int(round(fj)) - half, int(round(fj)) + half)

    # mask out indices that fall outside the grid
    i_mask = (i_win >= 0) & (i_win < grid.nx)
    j_mask = (j_win >= 0) & (j_win < grid.ny)

    # 1-D sinc weights
    wi = np.sinc(fi - i_win)
    wj = np.sinc(fj - j_win)

    W = np.zeros((grid.nx, grid.ny))
    W[np.ix_(i_win[i_mask], j_win[j_mask])] = np.outer(wi[i_mask], wj[j_mask])
    return W


def build_weight_matrix(grid, xy, n_objects, n_sinc: int) -> np.ndarray:
    """Precompute (nx*ny, n_sources) sinc injection weight matrix.
    The weight matrix is built such that multiplication by W encodes injection
    and multplication by W.T encodes extraction. This gives adjoint-safety for the
    inverse problem.
    """
    W = np.zeros((grid.nx * grid.ny, n_objects))

    for s in range(n_objects):
        x_s, y_s = xy[s]
        W[:, s] = _sinc_weights(grid, x_s, y_s, n_sinc).ravel()
    return W

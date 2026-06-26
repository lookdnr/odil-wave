import numpy as np
import math
from typing import Tuple

from odil_wave import Grid


def _place_ellipse(
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

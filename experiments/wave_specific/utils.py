from typing import Tuple
import numpy as np


def ray_receiver_locs(source_loc: Tuple, angles_deg: np.ndarray, radii: np.ndarray):
    """Generate receiver locations along rays from `source_loc`"""
    # extract src
    sx, sy = source_loc

    recv_locs = []

    for theta in angles_deg:
        rad = np.deg2rad(theta)  # convert to rad
        cos_t, sin_t = np.cos(rad), np.sin(rad)
        # compute loc by trig
        for r in radii:
            x = sx + r * cos_t
            y = sy + r * sin_t
            recv_locs.append((x, y))
    return recv_locs

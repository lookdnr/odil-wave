from typing import Tuple, List
import numpy as np


def ray_receiver_locs(source_loc: Tuple, angles_deg: List, radii: np.ndarray):
    """Generate receiver locations along rays from `source_loc`"""
    # extract src
    sx, sy = source_loc

    recv_locs = []
    rays = {}

    for theta in angles_deg:
        rad = np.deg2rad(theta)  # convert to rad
        cos_t, sin_t = np.cos(rad), np.sin(rad)
        per_ray_locs = []
        # compute loc by trig
        for r in radii:
            x = sx + r * cos_t
            y = sy + r * sin_t
            recv_locs.append((x, y))
            per_ray_locs.append((x, y))
        rays[theta] = per_ray_locs
    return recv_locs, rays

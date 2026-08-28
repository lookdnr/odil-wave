from odil_wave import Sources, Receivers
from .analytic import src_rec_distance

import numpy as np


def pre_reflection_mask(
    src: Sources, recvs: Receivers, times: np.ndarray, c: float
) -> np.ndarray:
    """Returns a boolean mask used to restrict the observations on traces
    to between the arrival and before the time at which spurious boundary
    oscillations might appear due to approximate ABCs.

    In a homogenous medium of wave speed c, a wave travels a distance r between
    a source and receiver in a time t_a = r / c (the arrival time).

    For a src with distance ds to the nearest wall and recv with distance dr to
    the same wall, we compute the lower bound reflection arrival time as
    (ds + dr) / c, the  time taken to travel the path src -> nearest wall and then
    nearest wall to src -> recv
    """
    # compute src-rec distances
    r = src_rec_distance(src, recvs)

    # extract source and rec locations
    sx, sy = np.asarray(src.src_xy).reshape(-1)[:2]
    rec_xy = np.asarray(recvs.recv_xy)

    # extrcat domain extent
    (xmin, xmax), (ymin, ymax) = src.grid.extent

    # distance from the source to each wall [left, right, bottom, top]
    d_src_walls = [sx - xmin, xmax - sx, sy - ymin, ymax - sy]

    # boolean mask in time for each recv
    mask = np.zeros((times.size, len(rec_xy)), dtype=bool)

    # fill the mask for each receiver: compute arrival time for the
    # direct arrival and the lower bound arrival time for the reflection
    # as the time taken for a wave to travel src -> wall -> recv,
    # ds + dr
    for k, (rx, ry) in enumerate(rec_xy):
        # compute distance of each recv to each wall
        d_rec_walls = [rx - xmin, xmax - rx, ry - ymin, ymax - ry]

        # compute min distance of rec from wall
        # this distance = distance of src from wall + distance of rec from wall
        d_reflect = min(ds + dr for ds, dr in zip(d_src_walls, d_rec_walls))

        # compute reflection and direct arrival times from distances
        t_reflect = d_reflect / c
        t_arrival = r[k] / c

        # mask between arrival and reflection times
        mask[:, k] = (times > t_arrival) & (times < t_reflect)

    return mask

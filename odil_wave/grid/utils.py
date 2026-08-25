import math


def points_per_wavelength(max_spacing: float, f0: float, c: float) -> float:
    """Compute the points per wavelength achieved by a grid.

    Parameters
    ----------
    max_spacing : float
        Coarsest grid spacing (max of dx, dy) [m].
    f0 : float
        Frequency to evaluate [Hz].
    c : float
        Wavespeed.

    Returns
    -------
    float
        Points per wavelength, lambda = c / f0, divided by `max_spacing`.
    """
    return (c / f0) / max_spacing


def nodes_for_ppw(length: float, f0: float, ppw: float, c: float) -> int:
    """Compute the smallest node count resolving a target Points per Wavelength (PPW).

    Parameters
    ----------
    length : float
        Physical length of the axis [m].
    f0 : float
        Frequency to resolve [Hz].
    ppw : float
        Minimum points per shortest wavelength (lambda = c / f0).
    c : float
        Wavespeed.

    Returns
    -------
    int
        Smallest node count achieving at least `ppw` points per wavelength.
    """
    lam = c / f0
    return math.ceil(ppw * length / lam) + 1

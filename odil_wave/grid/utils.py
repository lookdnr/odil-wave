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


def nyquist_cfl(dx: float, dy: float, c_max: float, f0: float) -> float:
    """Compute the cfl_safety at which dt exactly meets the temporal
    Nyquist limit for a source of frequency f0 (i.e., at dt = 1 / (2*f0)).

    Parameters
    ----------
    dx, dy : float
        Grid spacing along each axis (m).
    c_max : float
        Maximum wavespeed in the model (m/s).
    f0 : float
        Peak source frequency (Hz).

    Returns
    -------
    float
        cfl_safety value at which dt == 1 / (2*f0). Above this value,
        the time step under samples the source relative to Nyquist.
    """
    dt_cfl = 1.0 / (c_max * math.sqrt(1.0 / dx**2 + 1.0 / dy**2))
    dt_nyquist = 1.0 / (2.0 * f0)
    return dt_nyquist / dt_cfl

import numpy as np
from odil_wave import Wavefield


def create_u0(u0: Wavefield | np.ndarray | None = None, N: int = -1) -> np.ndarray:
    """Utility for creating flattened u0"""
    if u0 is None:
        u0 = np.zeros(N)
    elif isinstance(u0, Wavefield):
        u0 = u0.flat_data
    elif isinstance(u0, np.ndarray):
        u0 = np.asarray(u0).ravel()
    else:
        raise TypeError(
            "arg `u0` must be one of Wavefield, np.ndarray, None, got" + f" {type(u0)}"
        )
    return u0

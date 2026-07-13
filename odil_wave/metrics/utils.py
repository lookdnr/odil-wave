from odil_wave.wavefield import Wavefield

import numpy as np


def _decode_field(wf: Wavefield | np.ndarray) -> np.ndarray:
    """Helper to decode data"""
    if isinstance(wf, Wavefield):
        return wf.U
    elif isinstance(wf, np.ndarray):
        return wf
    else:
        raise TypeError(
            f"arg wf should be one of Wavefield, np.ndarray, got {type(wf)}"
        )


def _check_shape(u: np.ndarray, u_ref: np.ndarray) -> None:
    """Helper to assert two arrays have same shape"""
    assert np.array_equal(u.shape, u_ref.shape), (
        "args u and u_ref must have the same shape, "
        + f"got u.shape {u.shape} u_ref.shape {u_ref.shape}"
    )

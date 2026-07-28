from odil_wave.wavefield import Wavefield
from odil_wave.geometry import Sources
from odil_wave.operator import WaveEquation
from .utils import _decode_field, _check_shape

import numpy as np


def _decode_pair(
    u: Wavefield | np.ndarray, u_ref: Wavefield | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Decode both fields to arrays and assert matching shape."""
    u, u_ref = _decode_field(u), _decode_field(u_ref)
    _check_shape(u, u_ref)
    return u, u_ref


def _relative_norm(
    u: Wavefield | np.ndarray,
    u_ref: Wavefield | np.ndarray,
    idx: int | None = None,
    ord: float = 2.0,
) -> np.floating:
    """Helper to compute the `ord` norm of the difference between two wavefields at time
    level idx"""
    u, u_ref = _decode_pair(u, u_ref)

    _check_shape(u, u_ref)

    if idx is not None:
        u, u_ref = u[idx], u_ref[idx]

    u, u_ref = u.ravel(), u_ref.ravel()  # type: ignore

    return np.linalg.norm(u - u_ref, ord) / np.linalg.norm(u_ref, ord)


def relative_l2(
    u: Wavefield | np.ndarray, u_ref: Wavefield | np.ndarray
) -> np.floating:
    """Compute the L2 norm of the difference between an observed wavefield `u` and
    a reference wavefield `u_ref`.
    """
    return _relative_norm(u, u_ref, ord=2)


def final_time_l2(
    u: Wavefield | np.ndarray, u_ref: Wavefield | np.ndarray
) -> np.floating:
    """Compute the L2 norm of the difference between an observed wavefield `u` and
    a reference wavefield `u_ref` at the final time level.
    """
    return _relative_norm(u, u_ref, -1, ord=2)  # final time level


def relative_linfty(
    u: Wavefield | np.ndarray, u_ref: Wavefield | np.ndarray
) -> np.floating:
    """Compute the infinity norm of the difference between an observed wavefield `u` and
    a reference wavefield `u_ref`.
    """
    return _relative_norm(u, u_ref, ord=np.inf)


def l2_error_history(u: np.ndarray, u_ref: np.ndarray) -> np.ndarray:
    """Per time level L2 error, normalised by refernce norm"""
    u, u_ref = _decode_pair(u, u_ref)
    return np.linalg.norm(u - u_ref, axis=1) / np.linalg.norm(u_ref)


def residual(A: WaveEquation, u: Wavefield | np.ndarray, source: Sources):
    """Compute the the residual field r = Au - s"""
    u = _decode_field(u)
    s = source.source_matrix()[:, 0]
    return A.residual(u, s)


def residual_l2(A: WaveEquation, u: Wavefield | np.ndarray, source: Sources) -> float:
    """Compute the l2 norm of the residual Au - s"""
    return float(np.linalg.norm(residual(A, u, source), ord=2.0))


def residual_linfty(
    A: WaveEquation, u: Wavefield | np.ndarray, source: Sources
) -> float:
    """Infinity norm of the residual Au - s"""
    return float(np.linalg.norm(residual(A, u, source), ord=np.inf))


def relative_pde_residual(
    A: WaveEquation, u: Wavefield | np.ndarray, source: Sources
) -> float:
    """Residual norm ||Au - s|| relative to the source scale ||dt^2 s||."""
    r = residual(A, u, source)
    s = source.source_matrix()[:, 0]
    return float(np.linalg.norm(r) / np.linalg.norm(A.dt2 * s))

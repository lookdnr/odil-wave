from odil_wave.wavefield import Wavefield
from odil_wave.geometry import Sources
from odil_wave.operator import WaveEquation
from .utils import _decode_field, _check_shape

import numpy as np


def _relative_norm(
    u: Wavefield | np.ndarray,
    u_ref: Wavefield | np.ndarray,
    idx: int | None = None,
    ord: float = 2.0,
) -> np.floating:
    """Helper to compute the `ord` norm of the difference between two wavefields at time
    level idx"""
    u = _decode_field(u)
    u_ref = _decode_field(u_ref)

    _check_shape(u, u_ref)

    if idx is not None:
        u, u_ref = u[idx], u_ref[idx]

    u, u_ref = u.ravel(), u_ref.ravel()  # type: ignore

    return np.linalg.norm(u - u_ref, ord) / np.linalg.norm(u_ref, ord)


def residual_norm(
    A: WaveEquation, u: Wavefield | np.ndarray, source: Sources, ord: float = 2.0
):
    """Compute the `ord` norm of the residual Au - s"""
    u = _decode_field(u)
    s = source.source_matrix()[:, 0]
    r = A.residual(u, s)
    return np.linalg.norm(r, ord=ord)


def residual(A: WaveEquation, u: Wavefield | np.ndarray, source: Sources):
    """Compute the the residual field r = Au - s"""
    u = _decode_field(u)
    s = source.source_matrix()[:, 0]
    return A.residual(u, s)


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
    u = _decode_field(u)
    u_ref = _decode_field(u_ref)

    _check_shape(u, u_ref)

    return np.linalg.norm(u - u_ref, axis=1) / np.linalg.norm(u_ref)


def relative_pde_residual(u, wave_eq, f) -> float:
    r = wave_eq.residual(_decode_field(u).ravel(), f)
    return float(np.linalg.norm(r) / np.linalg.norm(wave_eq.dt2 * f))

import numpy as np

from odil_wave.geometry import Receivers
from odil_wave.wavefield import Wavefield
from .utils import _decode_field, _check_shape


def trace_misfit_norm(
    u: Wavefield | np.ndarray,
    u_ref: Wavefield | np.ndarray,
    receivers: Receivers,
    aggregate: bool = True,
):
    """Relative L2 misfit between receiver traces of `u` and `u_ref`."""
    u, u_ref = _decode_field(u), _decode_field(u_ref)
    _check_shape(u, u_ref)

    d_obs = receivers.extract_observations(u)  # (nt, n_recv)
    d_ref = receivers.extract_observations(u_ref)

    if not aggregate:
        return np.linalg.norm(d_obs - d_ref, axis=0) / np.linalg.norm(d_ref, axis=0)
    return np.linalg.norm(d_obs - d_ref) / np.linalg.norm(d_ref)


def trace_misfit(
    u: Wavefield | np.ndarray,
    u_ref: Wavefield | np.ndarray,
    receivers: Receivers,
):
    """Relative L2 misfit between receiver traces of `u` and `u_ref`."""
    u, u_ref = _decode_field(u), _decode_field(u_ref)
    _check_shape(u, u_ref)

    d_obs = receivers.extract_observations(u)  # (nt, n_recv)
    d_ref = receivers.extract_observations(u_ref)

    return d_obs - d_ref

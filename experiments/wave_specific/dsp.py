import numpy as np
import scipy.signal.windows as ssw
from typing import Tuple


def masked_taper(trace: np.ndarray, mask: np.ndarray, edge_frac=0.1) -> np.ndarray:
    """Zero a trace outside of 'mask' interval, and taper inside the interval
    to avoid hard truncation"""

    out = np.zeros_like(trace)
    idx = np.flatnonzero(mask)

    # too small
    if idx.size < 2:
        return out

    # start, end indices
    i0, i1 = idx[0], idx[-1] + 1
    win_len = i1 - i0

    # build Tukey window tapering within 2*edge_frac of the trace's edges
    taper = ssw.tukey(win_len, alpha=2 * edge_frac)

    out[i0:i1] = trace[i0:i1] * taper
    return out


def trace_spectra(
    traces: np.ndarray, dt: float, mask: np.ndarray, edge_frac=0.1
) -> Tuple[np.ndarray, ...]:
    """FFT each trace along a ray direction. Each trace is windowed and tapered
    to the pre-reflection window before FFT."""
    nt, n_recv = traces.shape

    # apply window and stack into array
    windowed = np.stack(
        [masked_taper(traces[:, k], mask[:, k], edge_frac) for k in range(n_recv)],
        axis=1,
    )

    # apply FFTs, extract angular frequencies
    ffts = np.fft.rfft(windowed, axis=0)
    omegas = np.fft.rfftfreq(nt, dt)  # nt samples, dt spacing
    return omegas, ffts


def pw_phase_velocity(
    omegas: np.ndarray,
    ffts: np.ndarray,
    radii: np.ndarray,
    c_bounds: Tuple[float, float] = (1450, 1550),
    n_trial: int = 2500,
) -> Tuple[np.ndarray, ...]:
    """slowness-frequency stack to compute the numerical phase velocity in
    accordance with
    Chekroun et al. Section 4: https://arxiv.org/pdf/1202.3427"""

    # set up trial slowness values across specified range
    c_min, c_max = min(c_bounds), max(c_bounds)
    c_grid = np.linspace(c_min, c_max, n_trial)

    # convert to angular frequency
    angular_f = 2 * np.pi * omegas

    # compute num modes, create stack output
    nf = len(angular_f)
    stack = np.zeros((nf, n_trial), dtype=complex)

    # compute the p-w stack quantity for each trial slowness
    for i, c_trial in enumerate(c_grid):
        p_trial = 1.0 / c_trial
        for k, omega in enumerate(angular_f):
            shift = np.exp(1j * omega * p_trial * radii)
            stack[k, i] = np.sum(ffts[k] * shift)  # apply shift and stack

    # compute stack magnitude
    mag = np.abs(stack)

    # extract max phase velocity
    idx = np.argmax(mag, axis=1)
    c_max = c_grid[idx].copy()

    return c_max, stack, mag

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

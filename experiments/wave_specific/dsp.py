import numpy as np
import scipy.signal.windows as ssw


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

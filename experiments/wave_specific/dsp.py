import numpy as np
import scipy.signal as ss
from typing import Dict


def xcorr_lags(ref: np.ndarray, test: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
    """Compute the cross correlation lags of a test trac against a reference"""
    corr = ss.correlate(test, ref, mode="full")
    lags_s = ss.correlation_lags(len(test), len(ref), mode="full") * dt
    return dict(corr=corr, lags=lags_s, peak=lags_s[np.argmax(corr)])


def envelopes(ref: np.ndarray, test: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute the envelopes of ref and test using the Hilbert transform"""
    return dict(
        ref=ss.envelope(ref), test=ss.envelope(test), ratio=test.max() / ref.max()
    )


def field_growth(U: np.ndarray, tail_frac: float = 0.3) -> Dict:
    """Compute per timestep max|u| and a late time growth rate slope
    tail_frac is the fraction of the trace (from the end) used for the growth-rate fit,
    meant to exclude the active injection period
    """
    max_u = np.abs(U).max(axis=1)
    nt = len(max_u)

    # extract tail end, 1-tail_frac from final time step
    tail_start = int(nt * (1 - tail_frac))
    tail = max_u[tail_start:]

    # detect infinite/ nan
    finite = bool(np.all(np.isfinite(U)))

    # filter
    k_full = np.arange(tail_start, nt)
    keep = tail > 0  # drop zero underflow samples
    if not finite or keep.sum() < 2:
        return dict(max_u=max_u.tolist(), growth_rate=float("nan"), finite=finite)

    # fit slope to approximate growth rate er time step
    slope, _ = np.polyfit(k_full[keep], np.log(tail[keep]), 1)
    return dict(max_u=max_u.tolist(), growth_rate=float(slope), finite=finite)

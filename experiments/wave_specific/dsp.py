import numpy as np
import scipy.signal as ss
from typing import Dict


def xcorr_lags(ref: np.ndarray, test: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
    """Compute the cross correlation lags of a test trac against a reference"""
    corr = ss.correlate(test, ref, mode="full")
    lags_s = ss.correlation_lags(len(test), len(ref), mode="full") * dt
    return dict(corr=corr, lags=lags_s, peak=lags_s[np.argmax(corr)])

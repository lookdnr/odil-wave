import numpy as np
import scipy.signal.windows as ssw
from typing import Tuple, Dict


def field_growth(U: np.ndarray, dt: float, tail_frac: float = 0.3) -> Dict:
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
        return dict(
            max_u=max_u.tolist(),
            growth_rate=float("nan"),
            r2=float("nan"),
            finite=finite,
        )

    # convert to units of time
    t_tail = k_full[keep] * dt
    log_tail = np.log(tail[keep])

    # fit slope to approximate growth rate er time step
    slope, intercept = np.polyfit(t_tail, log_tail, 1)

    pred = slope * t_tail + intercept  # compute prediction based on slope

    # compute R2 coeff
    ss_res = np.sum((log_tail - pred) ** 2)  # resid sum of sqr
    ss_tot = np.sum((log_tail - log_tail.mean()) ** 2)  # tot sum of sqr
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")  # r2

    return dict(max_u=max_u.tolist(), growth_rate=float(slope), r2=r2, finite=finite)


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
    c_bounds: Tuple[float, float] = (1300, 1600),
    n_trial: int = 7500,
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


def compute_attenuation(
    omegas: np.ndarray, ffts: np.ndarray, radii: np.ndarray
) -> np.ndarray:
    """Approximate per frequency attenuation coeff in accordance
    with Chekroun et al.

    Estimates the coefficient by a least squares fit between distance and
    amplitude spectra for each freq, determining the relationship between
    amp decay and distance

    Result is corrected for geometric spreading: 1/sqrt(r) term is removed
    """

    # take natural log
    log_amp = np.log(np.abs(ffts)) + 0.5 * np.log(radii)

    nf = len(omegas)
    alphas = np.empty(nf)

    # compute coeff for each freq
    for f in range(nf):
        slope, _ = np.polyfit(radii, log_amp[f], 1)
        alphas[f] = -slope

    return alphas


def analytical_phase_velocity(
    ppw: int, angle: float, c: float, dt: float, h: float
) -> float:
    """Solve the dispersion relation for angular frequency, return phase velocity
    C_h = w / k for a given ppw.

    Holds for second order in time and space, equation given by
    Alford, Kelly, and Boore: https://doi-org.iclibezp1.cc.ic.ac.uk/10.1190/1.1440470"""

    # compute wavelength and wavenumber
    wavelength = ppw * h  # assumes dx=dy=h
    wavenumber = 2 * np.pi / wavelength

    # convert angle to rad
    theta = np.deg2rad(angle)

    # compute constants in relation
    courant = c * dt / h
    kh_over_2 = wavenumber * h / 2

    # comptue rhs of relation
    rhs = courant**2 * (
        np.sin(kh_over_2 * np.cos(theta)) ** 2 + np.sin(kh_over_2 * np.sin(theta)) ** 2
    )

    # lhs is sin^2 w dt / 2
    # therefore w is 2/dt * arcsin(sqrt(rhs))
    omega = 2.0 / dt * np.arcsin(np.sqrt(rhs))
    return omega / wavenumber

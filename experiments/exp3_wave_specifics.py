import argparse
from dataclasses import replace, dataclass

import numpy as np

from common import (
    RunConfig,
    build_problem,
    run_reference,
    analytical_traces,
    run_optimiser,
)
from common.analytic import src_rec_distance
from common.compare import pre_reflection_mask
from wave_specific.dsp import xcorr_lags, envelopes
from odil_wave.grid.utils import nodes_for_ppw
from accuracy import save

SOURCE_LOC = (0.05, 0.05)
RECV_DISTANCES = (
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
)  # metres, along +x from the source
RECV_LOCS = tuple((SOURCE_LOC[0] + d, SOURCE_LOC[1]) for d in RECV_DISTANCES)
PPW_VALUES = [10, 15, 20, 25, 30, 35, 40]
BASE = RunConfig(
    nx=100,
    ny=100,  # overwritten per ppw below
    xmin=0.0,
    xmax=0.2,
    ymin=0.0,
    ymax=0.2,
    c_min=1500.0,
    c_max=1500.0,  # homogeneous
    cfl_safety=0.7,
    time_order=2,
    space_order=6,
    f0=100e3,
    source_loc=SOURCE_LOC,
    recv_mode="custom",
    recv_locs=RECV_LOCS,
    n_recvs=len(RECV_LOCS),
    model="homogeneous",
    method="paradiag",
    alpha=1e-3,
)


@dataclass
class DispersionResult:
    ppw: float
    nx: int
    nt: int
    distances: list
    odil: dict  # xcorrs, envs, slope, intercept
    devito: dict


def max_norm(trace: np.ndarray) -> np.ndarray:
    """Apply max normalisation to a trace"""
    return trace / trace.max()


def make_json_safe(d: dict):
    """Convert arrays to lists for json.dump"""
    return {
        key: (v.tolist() if isinstance(v, np.ndarray) else v) for key, v in d.items()
    }


def compute_correlations(
    d: np.ndarray, ana: np.ndarray, dt: float, mask: np.ndarray, distances: np.ndarray
):
    """Per receiver lag + envelope ratio against the analytic reference"""
    xcorrs, envs = [], []

    # apply pre rec mask, max norm, and compute for all
    for k in range(d.shape[1]):
        m = mask[:, k]
        obs, ref = d[m, k], ana[m, k]  # raw
        obs_n, ref_n = max_norm(obs), max_norm(ref)  # normalised
        xcorr = xcorr_lags(ref_n, obs_n, dt)
        env = envelopes(ref, obs)

        xcorrs.append(make_json_safe(xcorr))
        envs.append(make_json_safe(env))

    # extract peak lags
    lags = np.array([x["peak"] for x in xcorrs])

    # compute slope: fit a line with slope dr/dl
    slope, intercept = np.polyfit(distances, lags, 1)
    return xcorrs, envs, slope, intercept


def run(cfg, n_workers):
    """Colelct results for the given config"""
    c = cfg.c_min

    # build problem, run, get observations
    grid, _, src, recvs, _, _, opt = build_problem(cfg)
    res = run_optimiser(cfg, opt, n_workers=n_workers)
    solve_res = res.res
    d_odil, t_odil = recvs.extract_observations(solve_res.solution.U), grid.t

    # run devito, no need to account for JIT compile since we dc about time
    ref = run_reference(cfg, recvs.recv_xy)
    d_dev, t_dev = ref["traces"], ref["t"]

    # compute analytical traces and masks
    ana_odil = analytical_traces(src, recvs, t_odil, c)
    mask_odil = pre_reflection_mask(src, recvs, t_odil, c)
    ana_dev = analytical_traces(src, recvs, t_dev, c)
    mask_dev = pre_reflection_mask(src, recvs, t_dev, c)

    # compute src rec distances
    distances = src_rec_distance(src, recvs)

    xcorrs_o, envs_o, slope_o, intercept_o = compute_correlations(
        d_odil, ana_odil, grid.dt, mask_odil, distances
    )
    xcorrs_d, envs_d, slope_d, intercept_d = compute_correlations(
        d_dev, ana_dev, ref["dt"], mask_dev, distances
    )

    return DispersionResult(
        ppw=cfg.ppw,
        nx=cfg.nx,
        nt=grid.nt,
        distances=list(distances),
        odil=dict(xcorrs=xcorrs_o, envs=envs_o, slope=slope_o, intercept=intercept_o),
        devito=dict(xcorrs=xcorrs_d, envs=envs_d, slope=slope_d, intercept=intercept_d),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--dry", required=False, action="store_true")
    p.add_argument("--workers", required=False, default=32)
    a = p.parse_args()

    results = []
    for ppw in PPW_VALUES:
        # compute required nodes for given ppw and make problem
        n = nodes_for_ppw(BASE.xmax - BASE.xmin, BASE.f0, ppw, BASE.c_min)
        cfg = replace(BASE, nx=n, ny=n)

        # run
        if not a.dry:
            results.append(run(cfg, a.workers))

    # write
    if not a.dry:
        save(results, a.out)

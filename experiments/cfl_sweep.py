import argparse
import json
from dataclasses import replace

import numpy as np

from common import (
    RunConfig,
    build_problem,
    run_optimiser,
    run_reference,
    analytical_traces,
)
from common.compare import pre_reflection_mask
from odil_wave.metrics import normalised_trace_rel_l2
from wave_specific.dsp import field_growth
from odil_wave.grid.utils import nodes_for_ppw

XMAX = 0.1
F0 = 100e3
C_MIN = 1500.0
PPW = 10.0
N = nodes_for_ppw(XMAX, F0, PPW, C_MIN)

BASE = RunConfig(
    nx=N,
    ny=N,
    xmin=0.0,
    xmax=XMAX,
    ymin=0.0,
    ymax=XMAX,
    c_min=C_MIN,
    c_max=C_MIN,
    cfl_safety=0.7,  # overwritten per run
    time_order=2,
    space_order=2,  # 2nd order -> cfl_safety=1.0 is max
    f0=F0,
    source_loc=(XMAX / 2, XMAX / 2),
    recv_mode="custom",
    recv_locs=((XMAX * 0.8, XMAX / 2),),
    n_recvs=1,
    model="homogeneous",
    method="paradiag",
    alpha=1e-3,
    rtol=1e-8,
    allow_unstable=True,
)


def safe_run_devito(cfg, recv_xy, save):
    """Catch exceptions and flag non-finite output rather than letting one
    unstable config kill the whole sweep"""
    try:
        ref = run_reference(cfg, recv_xy, save=save)
    except Exception as e:
        return dict(traces=None, t=None, wall=float("nan"), finite=False, error=str(e))

    ref["finite"] = bool(np.all(np.isfinite(ref["traces"])))
    ref["error"] = None
    return ref


def run(cfl_safety: float, save: bool) -> dict:
    cfg = replace(BASE, cfl_safety=cfl_safety)
    grid, _, src, recvs, _, _, opt = build_problem(cfg)

    # ODIL
    res = run_optimiser(cfg, opt)
    U_odil = res.res.solution.U
    growth_odil = field_growth(U_odil)

    d_odil = recvs.extract_observations(U_odil)
    t_odil = grid.t

    ana_odil = analytical_traces(src, recvs, t_odil, cfg.c_min)
    mask_odil = pre_reflection_mask(src, recvs, t_odil, cfg.c_min)
    err_odil = float(normalised_trace_rel_l2(d_odil[mask_odil], ana_odil[mask_odil]))

    ref = safe_run_devito(cfg, recvs.recv_xy, save)
    if ref["traces"] is not None:
        d_dev, t_dev = ref["traces"], ref["t"]
        growth_dev = field_growth(d_dev)
        ana_dev = analytical_traces(src, recvs, t_dev, cfg.c_min)
        mask_dev = pre_reflection_mask(src, recvs, t_dev, cfg.c_min)
        err_dev = float(normalised_trace_rel_l2(d_dev[mask_dev], ana_dev[mask_dev]))

    else:
        growth_dev = dict(max_u=[], growth_rate=float("nan"), finite=False)
        err_dev = float("nan")

    return dict(
        cfl_safety=cfl_safety,
        nx=cfg.nx,
        nt=grid.nt,
        wall=res.wall,
        iters=res.iters,
        converged=res.converged,
        growth_rate_odil=growth_odil["growth_rate"],
        finite_odil=growth_odil["finite"],
        growth_rate_dev=growth_dev["growth_rate"],
        finite_dev=growth_dev["finite"],
        err_odil=err_odil,
        err_dev=err_dev,
        devito_error=ref["error"],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfl-safety", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--save", action="store_true", required=False)
    a = p.parse_args()

    row = run(a.cfl_safety, a.save)
    with open(a.out, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")

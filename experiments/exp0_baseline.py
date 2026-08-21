import argparse
from dataclasses import replace

from common import (
    RunConfig,
    build_problem,
    run_optimiser,
    analytical_traces,
    BaselineResult,
)
from common.compare import pre_reflection_mask
from odil_wave.metrics import normalised_trace_rel_l2
from accuracy.storage import save
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
    cfl_safety=0.7,
    time_order=2,
    space_order=6,
    f0=F0,
    source_loc=(XMAX / 2, XMAX / 2),  # mid domain
    recv_mode="custom",
    recv_locs=((XMAX * 0.8, XMAX / 2),),  # near edge
    n_recvs=1,
    model="homogeneous",
    alpha=1e-3,
    rtol=1e-8,
)

METHODS = ["paradiag", "gmres", "lbfgs"]


def run_one(method: str, maxiter: int) -> BaselineResult:
    """Run single config using specified method"""
    cfg = replace(BASE, method=method)
    grid, _, src, recvs, _, _, opt = build_problem(cfg)
    res = run_optimiser(cfg, opt, maxiter=maxiter)

    # extract obs and compute errors
    d = recvs.extract_observations(res.res.solution.U)
    t = grid.t

    ana = analytical_traces(src, recvs, t, cfg.c_min)
    mask = pre_reflection_mask(src, recvs, t, cfg.c_min)
    err = normalised_trace_rel_l2(d[mask], ana[mask])

    return BaselineResult(
        method=method,
        wall=res.wall,
        iters=res.iters,
        converged=res.converged,
        message=res.message,
        err=float(err),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--maxiter", type=int, default=500)  # cap, mainly for lbfgs
    p.add_argument("--out", required=False)
    a = p.parse_args()

    results = [run_one(m, a.maxiter) for m in METHODS]

    for r in results:
        print(
            f"{r.method:>10}: wall={r.wall:8.3f}s  iters={r.iters:5d}  "
            f"converged={r.converged!s:5}  err={r.err:.4f}  {r.message}"
        )

    if a.out:
        save(results, a.out)

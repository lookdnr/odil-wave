from common import RunConfig, build_problem, run_reference
from performance import peak_rss, repeat
from odil_wave.grid.utils import nodes_for_ppw

import argparse
import json
import os
from dataclasses import replace
from time import perf_counter

BASE = RunConfig(
    nx=100,
    ny=100,  # derived from f0/ppw at runtime
    xmin=0.0,
    xmax=0.2,
    ymin=0.0,
    ymax=0.2,
    c_min=1500.0,
    c_max=1800.0,
    cfl_safety=0.7,
    time_order=2,
    space_order=6,
    f0=50e3,
    source_loc=(0.10, 0.10),
    recv_mode="ring",
    n_recvs=8,  # perf study: traces don't matter
    ring_centre=(0.10, 0.10),
    a_frac=0.5,
    b_frac=0.5,
    model="inclusion",
    contrast=300.0,
    centre=(0.10, 0.10),
    radius=0.03,
    method="paradiag",
    alpha=1e-3,
)


def run_odil(cfg, ncores, caching, restart):
    grid, _, _, recvs, _, _, opt = build_problem(cfg)
    t0 = perf_counter()
    res = opt.minimise(
        method="paradiag",
        alpha=cfg.alpha,
        rtol=cfg.rtol,
        restart=restart,
        caching=caching,
        n_workers=ncores,
    )
    wall = perf_counter() - t0
    inner = res.recorder.outers[-1].inner
    return dict(
        wall=wall,
        t_setup=res.recorder.meta.get("t_setup"),
        t_solve=inner.t_solve,
        iters=inner.iters,
        n_matvecs=inner.n_matvecs,
        rho=float(inner.rho),
        residuals=list(inner.residual_history),
        nt=grid.nt,
        ns=grid.nx * grid.ny,
        dof=grid.nx * grid.ny * grid.nt,
        n_modes=(grid.nt - 2) // 2 + 1,
    )


def run_devito(cfg):
    grid, _, _, recvs, _, _, _ = build_problem(cfg)
    ref = run_reference(cfg, recvs.recv_xy)
    return dict(
        wall=ref["wall"],
        nt=grid.nt,
        ns=grid.nx * grid.ny,
        dof=grid.nx * grid.ny * grid.nt,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--solver", choices=["odil", "devito"], required=True)
    p.add_argument("--mode", choices=["cached", "uncached", "na"], default="na")
    p.add_argument("--ncores", type=int, required=True)
    p.add_argument("--f0", type=float, required=True)
    p.add_argument("--ppw", type=float, default=10.0)
    p.add_argument("--restart", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    # compute grid size for given f0
    n = nodes_for_ppw(BASE.xmax - BASE.xmin, a.f0, a.ppw, BASE.c_min)
    cfg = replace(BASE, nx=n, ny=n, f0=a.f0)

    # run
    if a.solver == "odil":
        fn, nw = (
            lambda: run_odil(cfg, a.ncores, a.mode == "cached", a.restart)
        ), a.ncores
    else:
        run_devito(cfg)  # warm-up: discard JIT compile
        fn, nw = (lambda: run_devito(cfg)), 0

    # get metrics over repeat runs
    metrics, stats = repeat(fn, a.repeats)
    peak, self_p, child_p = peak_rss(nw)

    row = dict(
        **metrics,
        **stats,
        solver=a.solver,
        mode=a.mode,
        ncores=a.ncores,
        nx=cfg.nx,
        f0=a.f0,
        ppw=a.ppw,
        restart=a.restart,
        peak_rss=peak,
        self_rss=self_p,
        child_rss=child_p,
        omp=os.environ.get("OMP_NUM_THREADS"),
    )
    with open(a.out, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")

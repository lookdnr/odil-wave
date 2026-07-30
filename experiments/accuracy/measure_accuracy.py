from time import perf_counter
import numpy as np

from common import RunConfig, build_problem, run_reference, analytical_traces
from common.compare import pre_reflection_mask
from odil_wave.metrics import normalised_trace_rel_l2


def measure_accuracy(cfg: RunConfig) -> dict:
    """
    Solve a problem with config `cfg` with ODIL and Devito.
    Score both against the analytic truth on the pre-reflection window"""
    c = cfg.c_min

    # ODIL setup
    grid, _, src, recvs, _, _, opt = build_problem(cfg)

    # measure solve time
    t0 = perf_counter()
    res = opt.minimise(method=cfg.method, alpha=cfg.alpha)
    wall_odil = perf_counter() - t0

    # extract ODIL observations at receivers
    d_odil = recvs.extract_observations(res.solution.U)
    t_odil = grid.t
    inner = res.recorder.outers[-1].inner
    iters = inner.iters if inner else res.nit  # inner GMRES count

    # Devito
    run_reference(cfg, recvs.recv_xy)  # run once so JIT compilation time ignored
    ref = run_reference(cfg, recvs.recv_xy)  # solve properly

    # extract traces, devito grid, and wall clock time
    d_dev, t_dev, wall_dev = ref["traces"], ref["t"], ref["wall"]

    # score both against the analytic truth on the window
    def windowed_err(d: np.ndarray, t: np.ndarray):
        d_ana = analytical_traces(src, recvs, t, c)  # compute traces on grid
        m = pre_reflection_mask(src, recvs, t, c)  # apply pre-reflection mask
        err = normalised_trace_rel_l2(
            d[m], d_ana[m]
        )  # compute norm of normalised trace
        return err

    return dict(
        nx=cfg.nx,
        nt=grid.nt,
        dof=grid.nx * grid.ny * grid.nt,
        dx=grid.dx,
        ppw=cfg.ppw,
        err_odil=windowed_err(d_odil, t_odil),  # error on odil grid
        err_dev=windowed_err(d_dev, t_dev),  # error on devito grid
        wall_odil=wall_odil,
        wall_dev=wall_dev,
        iters=iters,
    )

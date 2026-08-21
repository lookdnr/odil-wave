from common import (
    RunConfig,
    build_problem,
    run_reference,
    analytical_traces,
    run_optimiser,
    AccuracyResult,
    ReceiverReport,
)
from common.compare import pre_reflection_mask
from odil_wave.metrics import normalised_trace_rel_l2
from common.analytic import src_rec_distance

import numpy as np


def _per_receiver_err(d: np.ndarray, ana: np.ndarray, mask: np.ndarray):
    """Compute error for each receiver"""
    return [
        normalised_trace_rel_l2(d[mask[:, k], k], ana[mask[:, k], k])
        for k in range(d.shape[1])
    ]


def measure_accuracy(cfg: RunConfig) -> AccuracyResult:
    """
    Solve a problem with config `cfg` with ODIL and Devito.
    Score both against the analytic truth on the pre-reflection window"""
    c = cfg.c_min

    # ODIL setup
    grid, _, src, recvs, _, _, opt = build_problem(cfg)

    # measure ODIL solve time
    res = run_optimiser(cfg, opt)

    # extract results
    solve_res, wall_odil, iters = res.res, res.wall, res.iters

    d_odil = recvs.extract_observations(solve_res.solution.U)
    t_odil = grid.t

    # extract ODIL observations at receivers
    d_odil = recvs.extract_observations(solve_res.solution.U)
    t_odil = grid.t

    # Devito
    # measures internally
    ref = run_reference(cfg, recvs.recv_xy)  # solve properly

    # extract traces, devito grid, and wall clock time
    d_dev, t_dev, wall_dev = ref["traces"], ref["t"], ref["wall"]

    # truth + window on each solver's own axis
    ana_odil = analytical_traces(src, recvs, t_odil, c)
    mask_odil = pre_reflection_mask(src, recvs, t_odil, c)
    ana_dev = analytical_traces(src, recvs, t_dev, c)
    mask_dev = pre_reflection_mask(src, recvs, t_dev, c)

    # compute src-rec distances and report error for each receiver
    r = src_rec_distance(src, recvs)
    eo = _per_receiver_err(d_odil, ana_odil, mask_odil)
    ed = _per_receiver_err(d_dev, ana_dev, mask_dev)

    # generate report
    receivers = [
        ReceiverReport(k, recvs.recv_xy[k], float(r[k]), float(eo[k]), float(ed[k]))
        for k in range(recvs.n_receivers)
    ]

    # compute L2 norm of error for both
    err_odil = normalised_trace_rel_l2(d_odil[mask_odil], ana_odil[mask_odil])
    err_dev = normalised_trace_rel_l2(d_dev[mask_dev], ana_dev[mask_dev])

    # report metrics and traces
    metrics = dict(
        nx=cfg.nx,
        nt=grid.nt,
        dof=grid.nx * grid.ny * grid.nt,
        dx=grid.dx,
        ppw=cfg.ppw,
        err_odil=err_odil,
        err_dev=err_dev,
        wall_odil=wall_odil,
        wall_dev=wall_dev,
        iters=iters,
    )

    traces = dict(
        t_odil=t_odil,
        d_odil=d_odil,
        ana_odil=ana_odil,
        mask_odil=mask_odil,
        t_dev=t_dev,
        d_dev=d_dev,
        ana_dev=ana_dev,
        mask_dev=mask_dev,
    )
    return AccuracyResult(metrics, traces, receivers)


def measure_accuracy_repeated(cfg: RunConfig, n_repeats: int = 3):
    # warm start solvers
    _, _, _, recvs, _, _, opt = build_problem(cfg)
    _ = run_optimiser(cfg, opt)
    _ = run_reference(cfg, recvs.recv_xy)  # warm start devito

    # measure n_repeats runs
    runs = []
    for run in range(n_repeats):
        runs.append(measure_accuracy(cfg))
        print(runs[run].metrics, "\n")

    agg = {}  # aggregate results

    for k in runs[0].metrics:
        vals = np.array([r.metrics[k] for r in runs], float)
        agg[k] = float(vals.mean())  # report mean of each metric
        agg[f"{k}_std"] = float(vals.std(ddof=1))  # sample std

    result = runs[-1]  # keep one run's traces/receivers for plotting
    result.metrics = agg
    return result

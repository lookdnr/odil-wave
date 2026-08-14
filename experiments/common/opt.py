from .config import RunConfig
from time import perf_counter
from odil_wave import GaussNewtonOptimiser, LBFGSB

from dataclasses import dataclass
from odil_wave.metrics import SolveResult


@dataclass
class OptRunResult:
    res: SolveResult
    wall: float
    iters: int
    converged: bool
    message: str


def run_optimiser(
    cfg: RunConfig,
    opt: GaussNewtonOptimiser | LBFGSB,
    maxiter=2000,
    restart=10,
    n_workers=1,
) -> OptRunResult:
    """Shared util to run & time an optimisation"""

    t0 = perf_counter()
    if isinstance(opt, GaussNewtonOptimiser):
        res = opt.minimise(
            method=cfg.method,
            alpha=cfg.alpha,
            rtol=cfg.rtol,
            caching=cfg.caching,
            restart=restart,
            n_workers=n_workers,
        )
    else:  # LBFGSB
        res = opt.minimise(maxiter=maxiter, ftol=cfg.rtol, gtol=cfg.rtol)
    wall = perf_counter() - t0

    inner = res.recorder.outers[-1].inner
    iters = inner.iters if inner is not None else res.nit

    return OptRunResult(
        res=res, wall=wall, iters=iters, converged=res.success, message=res.message
    )

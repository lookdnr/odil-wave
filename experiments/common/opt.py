from .config import RunConfig
from time import perf_counter
from odil_wave import GaussNewtonOptimiser, LBFGSB


def run_optimiser(cfg: RunConfig, opt: GaussNewtonOptimiser | LBFGSB, maxiter=2000):
    """Shared util to run & time an optimisation"""

    t0 = perf_counter()
    if isinstance(opt, GaussNewtonOptimiser):
        res = opt.minimise(
            method=cfg.method,  # "paradiag" or "gmres"
            alpha=cfg.alpha,
            rtol=cfg.rtol,
            caching=cfg.caching,
        )
    else:  # LBFGSB
        res = opt.minimise(maxiter=maxiter, ftol=cfg.rtol, gtol=cfg.rtol)
    wall = perf_counter() - t0

    inner = res.recorder.outers[-1].inner
    iters = (
        inner.iters if inner is not None else res.nit
    )  # GN: GMRES iters; scipy: outer nit

    return dict(
        method=cfg.method,
        wall=wall,
        iters=iters,
        converged=res.success,
        message=res.message,
    )

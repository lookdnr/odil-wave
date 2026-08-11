import resource
import numpy as np
from odil_wave.optimisation.precond import AlphaCirculantPreconditioner


def peak_rss(n_workers, baseline=0):
    KB = 1024
    self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * KB
    child_rss = (
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * KB
    )  # max single child
    own = max(child_rss - baseline, 0)  # child's own allocations excluding parents RSS
    return self_rss + n_workers * own, self_rss, child_rss


def per_mode_factor_bytes(we, alpha=1e-3, dtype=np.complex128) -> np.ndarray:
    """Measure the memoru usage in bytes per mode in the preconditioner"""
    precond = AlphaCirculantPreconditioner(
        we.reduced_blocks, we.nt - 2, alpha=alpha, dtype=dtype, cache_factors=True
    )

    # measure size in bytes for each factorised mode
    itemsize = np.dtype(dtype).itemsize
    return np.array([precond._factorise_mode(zk).nnz * itemsize for zk in precond._z])


def analytic_memory(we, alpha=1e-3, dtype=np.complex128, n_workers=1) -> dict:
    """Predicted resident memory in bytes, for cached vs on-the-fly pathways"""
    per_mode = per_mode_factor_bytes(we, alpha, dtype)  # cached
    n_modes = len(per_mode)
    n_workers = min(n_workers, n_modes)  # as in the parallel precond

    return dict(
        n_modes=n_modes,
        per_mode_bytes=per_mode,
        cached_total_bytes=per_mode.sum(),  # cached workers factorise everything once
        uncached_peak_bytes=n_workers
        * per_mode.max(),  # non-caching factorises for each gmres matvec
    )

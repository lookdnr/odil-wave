import numpy as np
import pytest
import threadpoolctl

from odil_wave.optimisation import GaussNewtonOptimiser


@pytest.fixture(autouse=True)
def _pin_blas():
    """Pin the parent BLAS to one thread so forking workers is fork-safe."""
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        yield


# ===== correctness =====


@pytest.mark.parametrize("n_workers", [1, 2, 4])
def test_solve_matches_march(loss, n_workers):
    """The recovered wavefield matches the exact discrete solution (march),
    on both the serial (n_workers=1) and parallel paths"""
    opt = GaussNewtonOptimiser(loss, outer_maxiter=10)
    U = opt.minimise(method="paradiag", n_workers=n_workers).solution.U

    we = loss.problem.wave_eq
    s = loss.problem.sources[:, 0]
    U_march = we.march(s)

    rel = np.linalg.norm(U.ravel() - U_march.ravel()) / np.linalg.norm(U_march.ravel())
    assert rel < 1e-4, f"solve (n_workers={n_workers}) off exact by {rel:.2e}"


# ===== validation =====


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_n_workers_raises(loss, bad):
    opt = GaussNewtonOptimiser(loss, outer_maxiter=2)
    with pytest.raises(ValueError, match="n_workers"):
        opt.minimise(method="paradiag", n_workers=bad)

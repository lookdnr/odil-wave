import os

import numpy as np
import pytest
import threadpoolctl
from scipy.sparse.linalg import splu, gmres

from odil_wave.optimisation.precond import (
    AlphaCirculantPreconditioner,
    ParallelAlphaCirculantPreconditioner,
)


@pytest.fixture(autouse=True)
def _pin_blas():
    """Pin the parent BLAS to one thread so forking is safe"""
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        yield


@pytest.fixture
def parallel_pc(wave_eq):
    """Build parallel preconditioners and teardown after each test"""
    created = []

    def _make(n_workers, alpha=1e-3):
        M = ParallelAlphaCirculantPreconditioner.from_wave_equation(
            wave_eq, n_workers, alpha
        )
        created.append(M)
        return M

    yield _make
    for M in created:
        M.shutdown()


def true_solve(blocks, n, ns, b):
    """Exact solve of the reduced BTTB system by forward substitution"""
    B0, B1, B2 = blocks
    lu = splu(B0.tocsc())
    B = b.reshape(n, ns)
    U = np.zeros((n, ns))
    for m in range(n):
        rhs = B[m].copy()
        if m >= 1:
            rhs -= B1 @ U[m - 1]
        if m >= 2:
            rhs -= B2 @ U[m - 2]
        U[m] = lu.solve(rhs)
    return U.ravel()


# =====numerical correctness =====


@pytest.mark.parametrize("n_workers", [1, 4])
@pytest.mark.parametrize("alpha", [1e-3, 0.2])
@pytest.mark.parametrize("caching", [True, False])
def test_matches_serial(wave_eq, parallel_pc, n_workers, alpha, caching):
    """Parallel matvec must be numerically identical to the serial matvec."""
    M_ser = AlphaCirculantPreconditioner.from_wave_equation(
        wave_eq, alpha, cache_factors=caching
    )
    M_ser.dtype = np.complex128  # match the parallel complex128 for exact compare
    M_par = parallel_pc(n_workers, alpha)

    rng = np.random.default_rng(0)
    v = rng.standard_normal(M_ser.n * M_ser.ns)

    np.testing.assert_allclose(M_par.matvec(v), M_ser.matvec(v), atol=1e-10, rtol=1e-12)


def test_independent_of_worker_count(parallel_pc):
    """Partitioning across a different number of workers must not change the result."""
    M1 = parallel_pc(1)
    M2 = parallel_pc(2)
    rng = np.random.default_rng(1)
    v = rng.standard_normal(M1.n * M1.ns)
    np.testing.assert_allclose(M1.matvec(v), M2.matvec(v), atol=1e-12)


def test_gmres_convergence(wave_eq, parallel_pc):
    """Preconditioned GMRES with the parallel preconditioner converges fast/correctly"""
    A = wave_eq.reduced_operator()
    ntm2, ns = wave_eq.nt - 2, wave_eq.nx * wave_eq.ny
    M = parallel_pc(2, 1e-3)

    rng = np.random.default_rng(0)
    b = rng.standard_normal(ntm2 * ns)
    u_true = true_solve(wave_eq.reduced_blocks, ntm2, ns, b)

    iters = 0

    def count(_):
        nonlocal iters
        iters += 1

    u, info = gmres(
        A,
        b,
        M=M.as_linear_operator(),
        rtol=1e-10,
        maxiter=50,
        callback=count,
        callback_type="pr_norm",
    )
    assert info == 0, "GMRES did not converge"
    np.testing.assert_allclose(u, u_true, atol=1e-10)


# ===== pool management ====


def test_worker_cap_warns_and_caps(wave_eq):
    """Requesting more workers than cores/modes warns and caps the pool size."""
    with pytest.warns(UserWarning):
        M = ParallelAlphaCirculantPreconditioner.from_wave_equation(
            wave_eq, n_workers=10_000
        )

    try:
        n_cores = len(os.sched_getaffinity(0))
        n_modes = (wave_eq.nt - 2) // 2 + 1
        assert len(M._procs) == len(M._parts) == min(n_cores, n_modes)

    finally:
        M.shutdown()


def test_shutdown_terminates_workers(wave_eq):
    """After shutdown every worker process is dead (no leaks)"""
    M = ParallelAlphaCirculantPreconditioner.from_wave_equation(wave_eq, n_workers=2)
    procs = list(M._procs)

    assert all(p.is_alive() for p in procs)
    M.shutdown()
    assert not any(p.is_alive() for p in procs)


def test_context_manager_tears_down(wave_eq):
    """`with` shuts the pool down on block exit."""
    with ParallelAlphaCirculantPreconditioner.from_wave_equation(
        wave_eq, n_workers=2
    ) as M:
        procs = list(M._procs)
        assert all(p.is_alive() for p in procs)
    assert not any(p.is_alive() for p in procs)

from odil_wave.optimisation.precond import AlphaCirculantPreconditioner

import pytest
import numpy as np
from scipy.sparse.linalg import splu, gmres


@pytest.mark.parametrize("alpha", [1.0, 0.0, -1.0])
def test_invalid_alpha_raises(wave_eq, alpha):
    with pytest.raises(ValueError, match="alpha"):
        AlphaCirculantPreconditioner.from_wave_equation(wave_eq, alpha)


def dense_alpha_circulant(blocks, n, alpha):
    """Set a dense alpha circulant preconditioner"""
    S = np.diag(np.ones(n - 1), -1)
    S[0, -1] = alpha
    return sum(
        np.kron(np.linalg.matrix_power(S, j), Bj.toarray())
        for j, Bj in enumerate(blocks)
    )


@pytest.mark.parametrize("alpha", [1e-3, 1e-6, 0.5])
def test_correctness(wave_eq, alpha):
    """Asser than the taper, FFT diagonalisation, and LU solves are correct"""
    eq = wave_eq
    M = AlphaCirculantPreconditioner.from_wave_equation(eq, alpha)

    ntm2, ns = eq.nt - 2, eq.nx * eq.ny
    blocks = wave_eq.reduced_blocks
    P = dense_alpha_circulant(blocks, ntm2, alpha)

    # create random vectors
    rng = np.random.default_rng(0)
    w = rng.standard_normal(ntm2 * ns)
    b = rng.standard_normal(ntm2 * ns)

    # test precond * inv_precond(v) = v
    np.testing.assert_allclose(P @ M.matvec(w), w, atol=1e-10, rtol=1e-12)

    # test inv_precond(precond * v) = v
    np.testing.assert_allclose(M.matvec(P @ b), b, atol=1e-10, rtol=1e-12)


@pytest.mark.parametrize("alpha", [1e-3, 1e-6, 0.5])
def test_structual_correctness(wave_eq, alpha):
    """The precond matrix should differ only in its wrap terms that live
    in the first two block rows. Check this."""
    eq = wave_eq
    M = AlphaCirculantPreconditioner.from_wave_equation(eq, alpha)

    ntm2, ns = eq.nt - 2, eq.nx * eq.ny
    A = eq.reduced_operator()

    # create random vectors
    rng = np.random.default_rng(0)
    v = rng.standard_normal(ntm2 * ns)

    # if M is the inverse of A, AMv should equal v
    # since they differ in wrap around terms, the only non zeros
    # in the below r should be in those positions
    r = A @ M.matvec(v) - v

    head, tail = r[: 2 * ns], r[2 * ns :]

    np.testing.assert_allclose(tail, 0.0, atol=1e-10)
    assert np.linalg.norm(head, np.inf) > 1e-6, (
        f"corner residual vanished ({np.linalg.norm(head, np.inf):.2e}); "
        "test may be vacuous"
    )


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


def test_approximation_quality(wave_eq):
    """M^-1 approaches the true inverse as alpha -> 0 (O(alpha) corner term)"""
    eq = wave_eq
    ntm2, ns = eq.nt - 2, eq.nx * eq.ny

    rng = np.random.default_rng(0)
    b = rng.standard_normal(ntm2 * ns)

    u_true = true_solve(eq.reduced_blocks, ntm2, ns, b)

    errs = []
    for alpha in [1e-1, 1e-2, 1e-3]:
        M = AlphaCirculantPreconditioner.from_wave_equation(eq, alpha)
        err = np.linalg.norm(M.matvec(b) - u_true) / np.linalg.norm(u_true)
        errs.append(err)

    # corner term is O(alpha), so error must decrease monotonically
    assert errs[0] > errs[1] > errs[2], f"errors not decreasing with alpha: {errs}"


def test_gmres_convergence(wave_eq):
    """Preconditioned GMRES reaches the true solution in few iterations"""
    eq = wave_eq
    ntm2, ns = eq.nt - 2, eq.nx * eq.ny
    A = eq.reduced_operator()
    M = AlphaCirculantPreconditioner.from_wave_equation(eq, alpha=1e-3)

    rng = np.random.default_rng(0)
    b = rng.standard_normal(ntm2 * ns)
    u_true = true_solve(eq.reduced_blocks, ntm2, ns, b)

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
        callback_type="pr_norm",  # fires once per inner iteration
    )

    assert info == 0, "GMRES did not converge in 50 iterations"
    np.testing.assert_allclose(u, u_true, atol=1e-8 * np.linalg.norm(u_true, np.inf))
    assert iters <= 10, f"expected fast ParaDiag convergence, took {iters} iterations"

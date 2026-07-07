from odil_wave.optimisation.precond import AlphaCirculantPreconditioner

import pytest
import numpy as np


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

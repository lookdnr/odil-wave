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

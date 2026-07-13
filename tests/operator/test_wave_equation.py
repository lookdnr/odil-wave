import pytest
import numpy as np
from odil_wave import WaveEquation, Grid, Wavefield
from odil_wave.models.base import VelocityModel
from odil_wave import Sources


class SmoothModel(VelocityModel):
    """Smooth, non-constant test model"""

    def __init__(self, grid: Grid, background_c: float = 1.0):
        super().__init__(grid, background_c)
        self.name = "MMS Test Model"
        self.X, self.Y = self.grid.X, self.grid.Y
        self.c = self._build()

    def _build(self):
        """Smooth, non-constant profile"""
        return self.background_c + 0.3 * np.sin(np.pi * self.X) * np.cos(np.pi * self.Y)


# ===== Problem components =====
@pytest.fixture
def grid():
    return Grid(nx=10, ny=10)


@pytest.fixture
def wf(grid):
    return Wavefield(grid)


@pytest.fixture
def model(grid):
    return SmoothModel(grid)


@pytest.fixture
def w_eq(wf, model):
    return WaveEquation(wf, model)


@pytest.fixture
def s(grid):
    return Sources(grid, n_sources=1, source_locs=((0.5, 0.5),))


# ===== Test via Method of Manufactured Solutions ====
"""
Let u_exact(t, x, y) = t^2(x^2 + y^2)

then
    utt = 2(x^2 + y^2)
    nabla(u) = uxx + uyy = 4t^2
    => f = utt - c^2 nabla(u) = 2(x^2 + y^2) - 4 * c^2 * t^2

then Au should approximately equal dt2 * f at interior nodes
"""


def test_wave_eq_mms(grid, model, w_eq):
    nx, ny, nt = grid.nx, grid.ny, grid.nt
    X, Y, t = grid.X, grid.Y, grid.t

    # create (nt, nx, ny) grid of t^2(x^2 + y^2)
    U = (t[:, None, None] ** 2) * (X**2 + Y**2)[None]
    U = U.reshape(grid.nt, -1)  # reshape to expected (t, nx*ny) format

    # computes Au using matvec
    out = w_eq.matvec(U.ravel()).reshape(nt, nx, ny)

    # compute exact f scaled by dt2 (matches the dt2 scaling applied in matvec)
    c2 = model.c**2
    f = w_eq.dt2 * (2 * (X**2 + Y**2)[None] - 4 * c2[None] * (t[:, None, None] ** 2))

    # skip: IC rows 0-1, Higdon boundary nodes (i/j = 0 or -1), last time step
    slices = (slice(2, -1), slice(1, -1), slice(1, -1))

    np.testing.assert_allclose(out[slices], f[slices], rtol=1e-9, atol=1e-9)


# ===== Adjoint consistency =====


def test_adjoint_consistency(w_eq):
    """<Au, v> == <u, A^T v> for random u, v."""
    N = w_eq.nt * w_eq.nx * w_eq.ny
    rng = np.random.default_rng(42)
    u = rng.standard_normal(N)
    v = rng.standard_normal(N)
    lhs = np.dot(w_eq.matvec(u), v)
    rhs = np.dot(u, w_eq.rmatvec(v))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


def test_higdon_bc_adjoint(w_eq):
    """<bc.apply(U), Rb> == <U, bc.apply_transpose(R)> for each boundary."""
    nt, ns = w_eq.nt, w_eq.nx * w_eq.ny
    rng = np.random.default_rng(0)
    for bc in w_eq._bcs:
        U = rng.standard_normal((nt, ns))
        # residual non-zero only at this boundary's columns
        R = np.zeros((nt, ns))
        R[:, bc.bdry_cols] = rng.standard_normal((nt, len(bc.bdry_cols)))
        lhs = np.dot(bc.apply(U).ravel(), R[:, bc.bdry_cols].ravel())
        rhs = np.dot(U.ravel(), bc.apply_transpose(R).ravel())
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


# ===== Initial conditions =====


def test_IC(w_eq):
    nt, nx, ny = w_eq.nt, w_eq.nx, w_eq.ny
    ns = nx * ny

    rng = np.random.default_rng(0)
    u = rng.standard_normal(nt * ns)
    U = u.reshape(nt, ns)

    Au = w_eq.matvec(u).reshape(nt, ns)

    # IC rows are overwritten after Higdon, so they hold for all spatial nodes
    np.testing.assert_allclose(Au[0], U[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Au[1], U[1], rtol=1e-12, atol=1e-12)


# ===== Time marching =====


def test_march(w_eq, s):
    """Test time marching behaves as expected"""
    f = s.source_matrix()[:, 0]
    U = w_eq.march(f)
    r = w_eq.residual(U.ravel(), f).reshape(w_eq.nt, -1)[:-1]
    assert np.linalg.norm(r) / np.linalg.norm(w_eq.dt2 * f) < 1e-10

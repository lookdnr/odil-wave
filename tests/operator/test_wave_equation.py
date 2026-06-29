import pytest
import numpy as np
import scipy.sparse as sp
from odil_wave import WaveEquation, Grid, Wavefield
from odil_wave.models.base import VelocityModel


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


# ===== Test via Method of Manufatured Solutions ====
"""
Let u_exact(t, x, y) = t^2(x^2 + y^2)

then
    utt = 2(x^2 + y^2)
    nabla(u) = uxx + uyy = 4t^2
    => f = utt - c^2 nabla(u) = 2(x^2 + y^2) - 4 * c^2 * t^2

then Au should approximately equal f
"""


def test_wave_eq_mms(grid, model, w_eq):
    nx, ny, nt = grid.nx, grid.ny, grid.nt
    X, Y, t = grid.X, grid.Y, grid.t

    # create (nt, nx, ny) grid of t^2(x^2 + y^2)
    U = (t[:, None, None] ** 2) * (X**2 + Y**2)[None]
    U = U.reshape(grid.nt, -1)  # reshape to expected (t, nx*ny) format

    # computes Au using matvec
    # flatten input, reshape output for comparison
    out = w_eq.matvec(U.ravel()).reshape(nt, nx, ny)

    # compute exact f
    c2 = model.c**2
    f = w_eq.dt2 * (2 * (X**2 + Y**2)[None] - 4 * c2[None] * (t[:, None, None] ** 2))

    # create slices that skip IC rows and truncated edges
    # rows 0-1 are IC-overwritten, so only include 2:-1
    slices = (slice(2, -1), slice(1, -1), slice(1, -1))

    np.testing.assert_allclose(out[slices], f[slices], rtol=1e-9, atol=1e-9)


# ===== Matvec validation =====


@pytest.fixture
def A(w_eq):
    """Construct the full linear operator for the wave equation"""
    nt, nx, ny = w_eq.nt, w_eq.nx, w_eq.ny
    ns = nx * ny  # number of spatial points
    It, Is = sp.eye(nt), sp.eye(ns)

    # extract operator matrices
    Dt = w_eq._ut_op.Dt  # ut op
    Dtt = w_eq._utt_op.Dtt  # utt op
    C2L = w_eq.C2L  # Laplacian op
    S = w_eq.S  # damping matrix

    # assemble full A matrix
    # A = dt^2 * (Dtt + Dt + c^2 L)
    A = w_eq.dt2 * (sp.kron(Dtt, Is) + sp.kron(Dt, S) - sp.kron(It, C2L))

    # apply same ICs
    A[0:ns, :] = sp.kron(sp.eye(1, nt, 0), Is)  # row block t=0: identity → IC1
    A[ns : 2 * ns, :] = sp.kron(sp.eye(1, nt, 1), Is)  # row block t=1: identity → IC2
    return A.tocsr()


def test_matvec_equals_assembled_A(A, w_eq):
    """WaveEquation.matvec() applies the full wave equation operator
    A without ever forming it explicitly. This tests it is equivalent
    to computing Au using the explicitly formed A.
    """
    nt, nx, ny = w_eq.nt, w_eq.nx, w_eq.ny
    ns = nx * ny  # number of spatial points

    # evaluate for random input
    rng = np.random.default_rng(0)
    u = rng.standard_normal(nt * ns)

    # test matvec(u) ~~ Au for full A
    np.testing.assert_allclose(A @ u, w_eq.matvec(u), rtol=1e-10, atol=1e-10)


def test_rmatvec_equals_assembled_AT(A, w_eq):
    """WaveEquation.rmatvec() applies the transposed operator A^T without
    ever forming it explicitly. This tests it is equivalent to computing
    A.T v using the explicitly formed A.
    """
    nt, nx, ny = w_eq.nt, w_eq.nx, w_eq.ny
    ns = nx * ny  # number of spatial points

    # evaluate for random input
    rng = np.random.default_rng(0)
    v = rng.standard_normal(nt * ns)

    # test rmatvec(v) ~~ A^T v for full A
    np.testing.assert_allclose(A.T @ v, w_eq.rmatvec(v), rtol=1e-10, atol=1e-10)


# ===== Conditions (damping BC, IC) =====


def test_IC(w_eq):
    nt, nx, ny = w_eq.nt, w_eq.nx, w_eq.ny
    ns = nx * ny

    rng = np.random.default_rng(0)
    u = rng.standard_normal(nt * ns)
    U = u.reshape(nt, ns)  # same layout matvec uses

    Au = w_eq.matvec(u).reshape(nt, ns)

    # IC1: row 0 returns u(t=0)
    np.testing.assert_allclose(Au[0], U[0], rtol=1e-12, atol=1e-12)

    # IC2: row 1 returns u(t=1) unchanged
    np.testing.assert_allclose(Au[1], U[1], rtol=1e-12, atol=1e-12)


def test_IC_enforced_by_solve(A, w_eq):
    nt, ns = w_eq.nt, w_eq.nx * w_eq.ny
    rng = np.random.default_rng(0)
    f = rng.standard_normal(nt * ns)
    f[:ns] = 0.0  # IC1 rhs
    f[ns : 2 * ns] = 0.0  # IC2 rhs

    u = sp.linalg.spsolve(A, f).reshape(nt, ns)

    np.testing.assert_allclose(u[0], 0.0, atol=1e-10)  # u(t=0) = 0
    np.testing.assert_allclose(u[1], 0.0, atol=1e-10)  # u(t=1) = 0

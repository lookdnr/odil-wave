import pytest
import numpy as np
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
    return Grid(interior_shape=(10, 10), pml_width=0)


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
    f = 2 * (X**2 + Y**2)[None] - 4 * c2[None] * (t[:, None, None] ** 2)

    # create slices that skip IC rows and truncated edges
    # rows 0-1 are IC-overwritten, so only include 2:-1
    slices = (slice(2, -1), slice(1, -1), slice(1, -1))

    np.testing.assert_allclose(out[slices], f[slices], rtol=1e-9, atol=1e-9)

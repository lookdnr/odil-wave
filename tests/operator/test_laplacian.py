import numpy as np
import pytest
from odil_wave.operator import Laplacian
from odil_wave import Grid, Wavefield


@pytest.fixture
def wf():
    # asymmetric on purpose: distinct nx/ny and dx/dy
    grid = Grid(nx=11, ny=13)
    return Wavefield(grid)


def _set_spatial_field(wf, field2d):
    """Broadcast a (nx, ny) spatial field across all time rows."""
    flat = field2d.ravel()
    wf.U = np.broadcast_to(flat, wf.U.shape).copy()


def _interior(out_row, nx, ny, half):
    """Reshape a flattened spatial row to (nx, ny) and trim the boundary."""
    return out_row.reshape(nx, ny)[half:-half, half:-half]


# ===== Smoke tests =====

OPERATORS = [Laplacian]
VALID_ORDERS = [2, 4, 6, 8]
INVALID_ORDERS = [0, 1, 3, 5, -2]


@pytest.mark.parametrize("Operator", OPERATORS)
@pytest.mark.parametrize("ord", VALID_ORDERS)
def test_instantiates_at_valid_orders(wf, Operator, ord):
    op = Operator(wf, ord=ord)
    assert op.ord == ord


@pytest.mark.parametrize("Operator", OPERATORS)
@pytest.mark.parametrize("ord", INVALID_ORDERS)
def test_invalid_order_raises(wf, Operator, ord):
    with pytest.raises(ValueError, match="accuracy order"):
        Operator(wf, ord=ord)


# ===== Validity checking =====


def test_laplacian_of_constant_is_zero(wf):
    nx, ny = wf.grid.shape
    _set_spatial_field(wf, np.ones((nx, ny)))

    out = Laplacian(wf).apply(wf.U)  # (nt, nx*ny)
    assert np.allclose(_interior(out[0], nx, ny, half=1), 0.0, atol=1e-12)


# Polynomial exactness

CASES = [
    (o, p, q)
    for o in (2, 4, 6, 8)
    for p in range(o + 2)  # exact up to degree ord+1
    for q in range(o + 2)
]


@pytest.mark.parametrize("ord,p,q", CASES)
def test_laplacian_polynomial_exactness(wf, ord, p, q):
    nx, ny = wf.grid.shape
    half = ord // 2
    X, Y = wf.grid.X, wf.grid.Y

    _set_spatial_field(wf, X**p * Y**q)
    out = Laplacian(wf, ord=ord).apply(wf.U)

    fxx = p * (p - 1) * X ** (p - 2) * Y**q if p >= 2 else np.zeros_like(X)
    fyy = q * (q - 1) * X**p * Y ** (q - 2) if q >= 2 else np.zeros_like(X)
    expected = fxx + fyy

    np.testing.assert_allclose(
        _interior(out[0], nx, ny, half),
        expected[half:-half, half:-half],
        rtol=1e-7,
        atol=1e-9,
    )

import pytest
import numpy as np
from odil_wave.operator import FirstTimeDerivative, SecondTimeDerivative
from odil_wave import Grid, Wavefield

# test parameters
OPERATORS = [FirstTimeDerivative, SecondTimeDerivative]
VALID_ORDERS = [2, 4, 6, 8]
INVALID_ORDERS = [0, 1, 3, 5, -2]


# test wavefield
@pytest.fixture
def wf():
    grid = Grid(nx=10, ny=20)  # odd, distinct sizes
    return Wavefield(grid)


# ===== smoke tests =====


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


# ===== validity checking =====


@pytest.mark.parametrize("ord", VALID_ORDERS)
def test_first_derivative_of_constant_is_zero(wf, ord):
    wf.U = np.full_like(wf.U, 10)  # constant field
    out = FirstTimeDerivative(wf, ord=ord).apply(wf.U)
    half = ord // 2
    assert np.allclose(out[half:-half], 0.0, atol=1e-12)


@pytest.mark.parametrize("ord", VALID_ORDERS)
def test_first_derivative_of_linear(wf, ord):
    t = wf.grid.t[:, None]
    wf.U = np.broadcast_to(t, wf.U.shape).copy()  # u = t
    out = FirstTimeDerivative(wf, ord=ord).apply(wf.U)
    half = ord // 2
    assert np.allclose(out[half:-half], 1.0, atol=1e-10)


@pytest.mark.parametrize("ord", VALID_ORDERS)
def test_second_derivative_of_linear_is_zero(wf, ord):
    t = wf.grid.t[:, None]
    wf.U = np.broadcast_to(t, wf.U.shape).copy()  # u = t
    out = SecondTimeDerivative(wf, ord=ord).apply(wf.U)
    half = ord // 2
    assert np.allclose(out[half:-half], 0.0, atol=1e-10)


# ===== test polynomial exactness =====
# uses the fact that a central diff scheme of order k should be exact
# for a polynomial of degree k for first diff and k+1 for 2nd


def _set_poly_in_time(wf, k):
    """Fill wf.U with u(t) = t**k, constant across space. Returns t (nt,)."""
    t = wf.grid.t
    col = (t**k)[:, None]  # (nt, 1)
    wf.U = np.broadcast_to(col, wf.U.shape).copy()
    return t


def _broadcast(expected_t, shape):
    return np.broadcast_to(expected_t[:, None], shape)


# (ord, degree) pairs: 1st-deriv exact up to degree == ord
FIRST_CASES = [(o, k) for o in (2, 4, 6, 8) for k in range(o + 1)]
# 2nd-deriv exact up to degree == ord + 1
SECOND_CASES = [(o, k) for o in (2, 4, 6, 8) for k in range(o + 2)]


@pytest.mark.parametrize("ord,k", FIRST_CASES)
def test_first_derivative_polynomial_exactness(wf, ord, k):
    half = ord // 2
    t = _set_poly_in_time(wf, k)
    out = FirstTimeDerivative(wf, ord=ord).apply(wf.U)

    deriv = k * t ** (k - 1) if k >= 1 else np.zeros_like(t)  # d/dt t^k
    expected = _broadcast(deriv, wf.U.shape)

    np.testing.assert_allclose(
        out[half:-half], expected[half:-half], rtol=1e-7, atol=1e-9
    )


@pytest.mark.parametrize("ord,k", SECOND_CASES)
def test_second_derivative_polynomial_exactness(wf, ord, k):
    half = ord // 2
    t = _set_poly_in_time(wf, k)
    out = SecondTimeDerivative(wf, ord=ord).apply(wf.U)

    deriv = (
        k * (k - 1) * t ** (k - 2) if k >= 2 else np.zeros_like(t)
    )  # second derivative of t^k
    expected = _broadcast(deriv, wf.U.shape)

    np.testing.assert_allclose(
        out[half:-half], expected[half:-half], rtol=1e-7, atol=1e-9
    )

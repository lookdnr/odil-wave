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
    grid = Grid(interior_shape=(10, 20), pml_width=0)  # odd, distinct sizes
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


def test_first_derivative_of_constant_is_zero(wf):
    wf.U = np.full_like(wf.U, 10)  # constant field
    out = FirstTimeDerivative(wf).apply(wf.U)

    assert np.allclose(out[1:-1], 0.0, atol=1e-12)  # take interior


def test_second_derivative_of_linear_is_zero(wf):
    t = wf.grid.t[:, None]  # reshape to (nt, 1) so can be broadcasted
    wf.U = np.broadcast_to(
        t, wf.U.shape
    ).copy()  # set u = t (const in space, linear in time)
    out = SecondTimeDerivative(wf).apply(wf.U)  # should be zero

    assert np.allclose(out[1:-1], 0.0, atol=1e-10)

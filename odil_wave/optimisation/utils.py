import numpy as np
from scipy.sparse.linalg import LinearOperator
from odil_wave import Wavefield


def create_u0(u0: Wavefield | np.ndarray | None = None, N: int = -1) -> np.ndarray:
    """Create a flattened initial guess vector.

    Parameters
    ----------
    u0 : Wavefield, np.ndarray, or None, optional
        Initial guess, a zero vector of length `N` is used if None.
    N : int, optional
        Length of the zero vector used when `u0` is None.

    Returns
    -------
    np.ndarray
        Flattened (N,) initial guess.

    Raises
    ------
    TypeError
        If `u0` is not one of Wavefield, np.ndarray, or None.
    """
    if u0 is None:
        u0 = np.zeros(N)
    elif isinstance(u0, Wavefield):
        u0 = u0.flat_data
    elif isinstance(u0, np.ndarray):
        u0 = np.asarray(u0).ravel()
    else:
        raise TypeError(
            "arg `u0` must be one of Wavefield, np.ndarray, None, got" + f" {type(u0)}"
        )
    return u0


class CountedOperator(LinearOperator):
    """`scipy.sparse.linalg.LinearOperator` wrapper that counts `matvec` calls.

    Parameters
    ----------
    A : scipy.sparse.linalg.LinearOperator
        Operator to wrap and count calls to.

    Attributes
    ----------
    count : int
        Running count of matvec applications.
    """

    def __init__(self, A: LinearOperator) -> None:
        super().__init__(dtype=A.dtype, shape=A.shape)
        self._A = A
        self.count = 0

    def _matvec(self, x):
        self.count += 1
        return self._A @ x

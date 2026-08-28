from .base import SparseOperator
import scipy.sparse as sp
import numpy as np
from odil_wave.wavefield import Wavefield

from .stencils import STENCIL_COEFFS_1ST, STENCIL_COEFFS_2ND, STENCIL_OFFSETS


def _diff_matrix(derivative: int, ord: int, n: int, h: float) -> sp.csr_matrix:
    """Return the nxn order `ord` differentiation matrix that represents the action
    of the second spatial derivative.
    """
    # extract coeffs
    coeffs = STENCIL_COEFFS_1ST if derivative == 1 else STENCIL_COEFFS_2ND
    top_coeffs, bot_coeff = coeffs[ord]
    offsets = STENCIL_OFFSETS[ord]

    # square h for 2nd derivative
    denom = h**2 if derivative == 2 else h

    # construct differentiation matrix
    D = sp.diags(
        diagonals=top_coeffs,
        offsets=offsets,  # type: ignore
        shape=(n, n),
        format="csr",
    ) / (bot_coeff * denom)
    return D


class FirstTimeDerivative(SparseOperator):
    """First time derivative operator, Dt kron Ixy.

    Parameters
    ----------
    wavefield : Wavefield
        Wavefield the operator acts on.
    ord : {2, 4, 6, 8}, optional
        Finite-difference accuracy order.

    Attributes
    ----------
    Dt : scipy.sparse.csr_matrix
        (nt, nt) first time derivative operator.
    """

    def __init__(self, wavefield: Wavefield, ord: int = 2) -> None:
        super().__init__(wavefield, ord)
        self.assemble()

    def assemble(self) -> None:
        """Assembles the time operator: Dt otimes Ixy"""
        ord = self.ord
        nt, dt = self.wavefield.grid.nt, self.wavefield.grid.dt

        self.Dt = _diff_matrix(derivative=1, ord=ord, n=nt, h=dt)

    def apply(self, U: np.ndarray) -> np.ndarray:
        """Apply the first time derivative operator to a (time, space) array.

        Parameters
        ----------
        U : np.ndarray
            (nt, nx*ny) array.

        Returns
        -------
        np.ndarray
            (nt, nx*ny) time derivative of `U`.
        """
        return self.Dt @ U

    def apply_transpose(self, U: np.ndarray) -> np.ndarray:
        """Apply the transposed first time derivative operator.

        Parameters
        ----------
        U : np.ndarray
            (nt, nx*ny) array.

        Returns
        -------
        np.ndarray
            (nt, nx*ny) result of Dt.T @ U.
        """
        return self.Dt.T @ U


class SecondTimeDerivative(SparseOperator):
    """Second time derivative operator, Dtt kron Ixy.

    Parameters
    ----------
    wavefield : Wavefield
        Wavefield the operator acts on.
    ord : {2, 4, 6, 8}, optional
        Finite-difference accuracy order.

    Attributes
    ----------
    Dt : scipy.sparse.csr_matrix
        (nt, nt) second time derivative operator.
    """

    def __init__(self, wavefield: Wavefield, ord: int = 2) -> None:
        super().__init__(wavefield, ord)
        self.assemble()

    def assemble(self) -> None:
        """Assembles the time operator: Dt otimes Ixy"""
        ord = self.ord
        nt, dt = self.wavefield.grid.nt, self.wavefield.grid.dt

        self.Dtt = _diff_matrix(derivative=2, ord=ord, n=nt, h=dt)

    def apply(self, U: np.ndarray) -> np.ndarray:
        """Apply the second time derivative operator to a (time, space) array.

        Parameters
        ----------
        U : np.ndarray
            (nt, nx*ny) array.

        Returns
        -------
        np.ndarray
            (nt, nx*ny) time derivative of `U`.
        """
        return self.Dtt @ U

    def apply_transpose(self, U: np.ndarray) -> np.ndarray:
        """Apply the transposed second time derivative operator.

        Parameters
        ----------
        U : np.ndarray
            (nt, nx*ny) array.

        Returns
        -------
        np.ndarray
            (nt, nx*ny) result of Dtt.T @ U.
        """
        return self.Dtt.T @ U

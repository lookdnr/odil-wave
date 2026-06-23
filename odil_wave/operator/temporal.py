from .base import SparseOperator
import scipy.sparse as sp
from odil_wave.wavefield import Wavefield

from .stencils import STENCIL_COEFFS_1ST, STENCIL_COEFFS_2ND, STENCIL_OFFSETS


def _diff_matrix(derivative: int, ord: int, n: int, h: float) -> sp.dia_matrix:
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
    """First derivative operator"""

    def __init__(self, wavefield: Wavefield, ord: int = 2) -> None:
        super().__init__(wavefield, ord)
        self.assemble()

    def assemble(self) -> None:
        """Assembles the time operator: Dt otimes Ixy"""
        ord = self.ord
        nt, dt = self.wavefield.grid.nt, self.wavefield.grid.dt

        self.Dt = _diff_matrix(derivative=1, ord=ord, n=nt, h=dt)


class SecondTimeDerivative(SparseOperator):
    """Second derivative operator"""

    def __init__(self, wavefield: Wavefield, ord: int = 2) -> None:
        super().__init__(wavefield, ord)
        self.matrix = self.assemble()

    def assemble(self) -> None:
        """Assembles the time operator: Dt otimes Ixy"""
        ord = self.ord
        nt, dt = self.wavefield.grid.nt, self.wavefield.grid.dt

        self.Dtt = _diff_matrix(derivative=2, ord=ord, n=nt, h=dt)

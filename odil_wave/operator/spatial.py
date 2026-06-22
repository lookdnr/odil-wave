from odil_wave.wavefield import Wavefield
from .base import SparseOperator
import scipy.sparse as sp
import numpy as np

from .stencils import STENCIL_OFFSETS, STENCIL_COEFFS_2ND


def _diff_matrix(ord: int, n: int, h: float) -> sp.dia_matrix:
    """Return the nxn order `ord` differentiation matrix that represents the action
    of the second spatial derivative.
    """
    top_coeffs, bot_coeff = STENCIL_COEFFS_2ND[ord]
    offsets = STENCIL_OFFSETS[ord]

    D = sp.diags(
        diagonals=np.array(top_coeffs),
        offsets=offsets,  # type: ignore
        shape=(n, n),
        format="csr",
    ) / (bot_coeff * h**2)
    return D


class Laplacian(SparseOperator):
    """2D Laplacian Operator"""

    def __init__(self, wavefield: Wavefield, ord: int = 2) -> None:
        super().__init__(wavefield, ord)
        self.matrix = self.assemble()

    def assemble(self) -> sp.dia_matrix:
        """Assembles the 2D Laplacian operator: Dxx otimes Iy + Ix otimes Dyy"""

        # extract parameters
        ord = self.ord
        nx, dx = self.wavefield.grid.nx, self.wavefield.grid.dx
        ny, dy = self.wavefield.grid.ny, self.wavefield.grid.dy

        # construct differentiation matrics
        Dxx = _diff_matrix(ord, n=nx, h=dx)
        Dyy = _diff_matrix(ord, n=ny, h=dy)

        # construct the identity matrices for the Kronecker products
        # Iy is ny*ny, Ix is nx*nx
        Iy = sp.eye(ny)
        Ix = sp.eye(nx)

        # construct the operator as a Kronecker sum
        return sp.kron(Dxx, Iy) + sp.kron(Ix, Dyy)

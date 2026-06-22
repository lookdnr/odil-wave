from odil_wave.wavefield import Wavefield
from .base import SparseOperator
import scipy.sparse as sp
import numpy as np

# hardcoded central difference stencil coefficients (2nd derivative)
# e.g. order : ([numerator coeffs], denom coeff)
STENCIL_COEFFS = {
    2: ([1.0, -2.0, 1.0], 1.0),
    4: ([-1.0, 16.0, -30.0, 16.0, -1.0], 12.0),
    6: ([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0], 180.0),
    8: ([-9.0, 128.0, -1008.0, 8064.0, 14350.0, 8064.0, -1008.0, 128.0, -9.0], 5040.0),
}

# global offsets for constructing matrix operators
# e.g., order : offsets
STENCIL_OFFSETS = {
    2: [-1, 0, 1],
    4: [-2, -1, 0, 1, 2],
    6: [-3, -2, -1, 0, 1, 2, 3],
    8: [-4, -3, -2, -1, 0, 1, 2, 3, 4],
}


def _diff_matrix(ord: int, n: int, h: float) -> sp.dia_matrix:
    """Return the nxn order `ord` differentiation matrix that represents the action
    of the second spatial derivative.
    """
    top_coeffs, bot_coeff = STENCIL_COEFFS[ord]
    offsets = STENCIL_OFFSETS[ord]

    D = sp.diags(
        diagonals=np.array(top_coeffs), offsets=offsets, shape=(n, n), format="csr"
    ) / (bot_coeff * h**2)
    return D


class SparseLaplacian(SparseOperator):
    """2D Laplacian Operator"""

    def __init__(self, wavefield: Wavefield, ord: int = 2) -> None:
        super().__init__(wavefield, ord)
        self.matrix = self.assemble()

    def assemble(self) -> sp.dia_matrix:
        """Assembles the 2D Laplacian operator: Iy otimes Dx + Dy otimes Ix"""

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
        return sp.kron(Iy, Dxx) + sp.kron(Dyy, Ix)

from odil_wave.wavefield import Wavefield
from .base import SparseOperator
from .ghost import GhostFill
import scipy.sparse as sp
import numpy as np

from .stencils import STENCIL_OFFSETS, STENCIL_COEFFS_2ND


def _diff_matrix(ord: int, n: int, h: float, ghost_width: int = 0) -> sp.csr_matrix:
    """Return the n x n+2g differentiation matrix of order `ord` that represents the
    action of the second spatial derivative.
    """
    top_coeffs, bot_coeff = STENCIL_COEFFS_2ND[ord]
    offsets = STENCIL_OFFSETS[ord]

    D = sp.diags(
        diagonals=np.array(top_coeffs),
        offsets=[o + ghost_width for o in offsets],  # type: ignore
        shape=(n, n + 2 * ghost_width),
        format="csr",
    ) / (bot_coeff * h**2)
    return D


class Laplacian(SparseOperator):
    """2D Laplacian Operator"""

    L: sp.csr_matrix
    Dxx: sp.csr_matrix  # second x derivative
    Dyy: sp.csr_matrix  # second y derivative

    def __init__(self, wavefield: Wavefield, ord: int = 2) -> None:
        super().__init__(wavefield, ord)
        self.L = self.assemble()

    def assemble(self) -> sp.csr_matrix:
        """Assembles the 2D Laplacian operator: Dxx otimes Iy + Ix otimes Dyy"""

        # extract parameters
        ord = self.ord
        nx, dx = self.wavefield.grid.nx, self.wavefield.grid.dx
        ny, dy = self.wavefield.grid.ny, self.wavefield.grid.dy
        g = ord // 2  # number of ghost nodes per side

        # construct differentiation matrics
        Dxx = _diff_matrix(ord, n=nx, h=dx, ghost_width=g)  # (nx, nx + 2g)
        Dyy = _diff_matrix(ord, n=ny, h=dy, ghost_width=g)  # (ny, ny + 2g)

        # store for use in Higdon BC
        self.Dxx = Dxx
        self.Dyy = Dyy

        # ghost fill operators
        Gx = GhostFill(nx, g).expand_x(ny)  # (ny*(nx + 2g), ny*ny)
        Gy = GhostFill(ny, g).expand_y(nx)  # (nx*ny, nx*(ny + 2g))

        # construct the identity matrices for the Kronecker products
        # Iy is ny*ny, Ix is nx*nx
        Iy = sp.eye(ny)
        Ix = sp.eye(nx)

        # kronecker products
        Dxx_kron = sp.kron(Dxx, Iy)
        Dyy_kron = sp.kron(Ix, Dyy)

        # construct the operator as a Kronecker sum
        # construct final operator as d^2/dx^2 with x ghost support
        # + d^2/dy^2 wiht y-ghost support
        return sp.csr_matrix(Dxx_kron @ Gx + Dyy_kron @ Gy)  # (nx*nx, ny*ny)

    def apply(self, U: np.ndarray) -> np.ndarray:
        """Apply operator to a (time, space) ndarray"""
        return U @ self.L.T

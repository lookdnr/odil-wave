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


def _assemble(
    deriv_ord: int, acc_ord: int, nt: int, nx: int, ny: int, dt: float
) -> sp.csr_matrix:
    """Helper for assembling the sparse deriv_ord derivative operator"""
    D = _diff_matrix(deriv_ord, acc_ord, nt, dt)
    Ixy = sp.eye(nx * ny)
    return sp.kron(D, Ixy).tocsr()  # type: ignore


class FirstTimeDerivative(SparseOperator):
    """First derivative operator"""

    def __init__(self, wavefield: Wavefield, ord: int = 2) -> None:
        super().__init__(wavefield, ord)
        self.matrix = self.assemble()

    def assemble(self) -> sp.csr_matrix:
        """Assembles the time operator: Dt otimes Ixy"""
        ord = self.ord
        nt, dt = self.wavefield.grid.nt, self.wavefield.grid.dt
        nx, _ = self.wavefield.grid.nx, self.wavefield.grid.dx
        ny, _ = self.wavefield.grid.ny, self.wavefield.grid.dy

        D = _assemble(1, ord, nt, nx, ny, dt)

        # enforce zero amplitude IC
        spatial_extent = nx * ny
        D[:spatial_extent, :] = sp.eye(nt * spatial_extent).tocsr()[:spatial_extent, :]
        return D


class SecondTimeDerivative(SparseOperator):
    """Second derivative operator"""

    def __init__(self, wavefield: Wavefield, ord: int = 2) -> None:
        super().__init__(wavefield, ord)
        self.matrix = self.assemble()

    def assemble(self) -> sp.csr_matrix:
        """Assembles the time operator: Dt otimes Ixy"""
        ord = self.ord
        nt, dt = self.wavefield.grid.nt, self.wavefield.grid.dt
        nx, _ = self.wavefield.grid.nx, self.wavefield.grid.dx
        ny, _ = self.wavefield.grid.ny, self.wavefield.grid.dy

        D = _assemble(2, ord, nt, nx, ny, dt)  # type: ignore
        spatial_extent = nx * ny

        # enforce zero amplitude IC
        D[:spatial_extent, :] = sp.eye(nt * spatial_extent).tocsr()[:spatial_extent, :]

        # enforce zero velocity IC
        ut = FirstTimeDerivative(self.wavefield, ord)
        ut_0 = ut.matrix[:spatial_extent, :]
        D[spatial_extent : 2 * spatial_extent, :] = ut_0

        # ensure we return a scipy.sparse.csr_matrix (not a csr_array)
        return D

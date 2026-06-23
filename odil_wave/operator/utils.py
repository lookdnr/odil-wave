from dataclasses import dataclass, field

from odil_wave.wavefield import Wavefield
from .temporal import FirstTimeDerivative, SecondTimeDerivative
from .spatial import Laplacian
from odil_wave.models.base import VelocityModel

import numpy as np
import scipy.sparse as sp


@dataclass
class WaveEquation:
    """Discrete acoustic wave equation u_tt - c^2*lap(u) = f."""

    wavefield: Wavefield
    model: VelocityModel
    time_order: int = 2
    space_order: int = 2

    _ut_op: FirstTimeDerivative = field(init=False)
    _utt_op: SecondTimeDerivative = field(init=False)
    _lap: Laplacian = field(init=False)

    def __post_init__(self):
        self._ut_op = FirstTimeDerivative(self.wavefield, self.time_order)
        self._utt_op = SecondTimeDerivative(self.wavefield, self.time_order)
        self._lap_op = Laplacian(self.wavefield, self.space_order)

        self.matrix = self.assemble()

    def assemble(self) -> sp.csr_matrix:
        grid = self.wavefield.grid
        nt, nxy = grid.nt, grid.nx * grid.ny

        # extrcat matrices for each operator
        D_t = self._ut_op.matrix
        D_tt = self._utt_op.matrix
        D_lap = self._lap_op.matrix

        I_t = sp.eye(nt, format="csr")  # (nt, nt) identity
        I_xy = sp.eye(nxy, format="csr")  # (nxy, nxy) identity

        c_sqr = sp.diags(self.model.model.ravel() ** 2)  # create C^2 matrix for mult
        Sigma = grid.sig_mat

        # assemble global matrix operator
        # notes:
        # - kron(D_tt, I_xy) gives time derivative at all points
        # - kron(D_t, Sigma) scales velocity across space bydiagonal entries
        # - kron(I_t, c^2*lap) gives lap across all time steps
        A = sp.csr_matrix(
            sp.kron(D_tt, I_xy) + sp.kron(D_t, Sigma) - sp.kron(I_t, c_sqr @ D_lap)
        )

        # enforce BCs

        # IC 1: first Nxy rows are identity
        u0 = sp.eye(nt * nxy, format="csr").tocsr()[:nxy, :]

        # IC 2: Ut(0) = 0, next nxy rows are t=0 rows of kron(Dt, Ixy)
        D_t_full = sp.kron(D_t, I_xy)
        ut0 = D_t_full[nxy : 2 * nxy, :]

        # stack IC rows into global operator
        A = sp.csr_matrix(sp.vstack([u0, ut0, A[2 * nxy :, :]], format="csr"))
        return A

    def residual(
        self, wavefield: Wavefield, c: VelocityModel, source: np.ndarray
    ) -> np.ndarray:
        """Compute Au - f, where A encodes the derivatives and PML condition"""
        # compute pde residual u_tt - c^2 u_xx
        pde = self._utt_op.apply(wavefield) - c.model**2 @ self._lap_op.apply(wavefield)

        # compute PML term Sigma*Ut
        pml = self.wavefield.grid.sig_mat @ self._ut_op.apply(wavefield)

        # compute residual: Utt + Sigma Ut - c^2(Uxx + Uyy) - f
        residual = pde + pml - source
        return residual

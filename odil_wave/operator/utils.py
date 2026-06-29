from dataclasses import dataclass, field

from odil_wave.wavefield import Wavefield
from .temporal import SecondTimeDerivative
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

    _utt_op: SecondTimeDerivative = field(init=False)
    _lap_op: Laplacian = field(init=False)

    C2L: sp.csr_matrix = field(init=False)  # c^2 * laplacian

    def __post_init__(self):
        self._utt_op = SecondTimeDerivative(self.wavefield, self.time_order)
        self._lap_op = Laplacian(self.wavefield, self.space_order)

        # precompute
        c_sqr = sp.diags(self.model.c.ravel() ** 2)
        self.C2L = c_sqr @ self._lap_op.L

        self.nt = self.wavefield.grid.nt
        self.nx, self.ny = self.wavefield.grid.shape

        # row scaling: multiplying the PDE block by dt^2 brings every
        # term to O(1) (dt^2 * c^2 / dx^2 = c^2 * CFL^2),
        # this balances the least-squares system without changing its solution
        self.dt2 = self.wavefield.grid.dt**2

    def matvec(self, u: np.ndarray) -> np.ndarray:
        """Compute the matrix vector product Au (no explicit A formation)"""
        U = u.reshape(self.nt, self.nx * self.ny)

        utt = self._utt_op.apply(U)
        lap = U @ self.C2L.T

        # apply dt**2 scaling
        AU = self.dt2 * (utt - lap)

        # enforce ICs
        # IC1: u(0) = 0
        AU[0, :] = U[0, :]

        # IC2: ut(0) = 0 - U[1] = U[0]  (1st-order forward diff from t=0)
        AU[1, :] = U[1, :]

        return AU.ravel()

    def rmatvec(self, r: np.ndarray) -> np.ndarray:
        """Compute the transposed matrix vector product A^T r
        (no explicit A formation)"""
        R = r.reshape(self.nt, self.nx * self.ny)

        Rz = R.copy()
        Rz[0, :] = 0.0  # adjoint of overwriting output rows 0,1:
        Rz[1, :] = 0.0  # the PDE terms must not see R[0], R[1]

        utt_t = self._utt_op.apply_transpose(Rz)  # Dtt.T @ R

        lap_t = Rz @ self.C2L

        ATv = self.dt2 * (utt_t - lap_t)

        # transpose of IC constraints
        ATv[0, :] += R[0, :]
        ATv[1, :] += R[1, :]

        return ATv.ravel()

    def residual(self, u: np.ndarray, f: np.ndarray) -> np.ndarray:
        """Compute Au - f, where A encodes the derivatives and PML condition

        Note that sources may be a (n_txy * n_shots) matrix encoding each of the shots
        """
        # apply dt**2 scaling to source term
        F = f.reshape(self.nt, self.nx * self.ny)
        Ff = self.dt2 * F
        Ff[0, :] = F[0, :]
        Ff[1, :] = F[1, :]
        return self.matvec(u) - Ff.ravel()

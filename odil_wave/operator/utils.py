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
    _lap_op: Laplacian = field(init=False)

    # operator components - we store these instead of assembling the full A
    S: sp.dia_matrix = field(init=False)  # damping coefficient matrix
    C2L: sp.csr_matrix = field(init=False)  # c^2 * laplacian

    def __post_init__(self):
        self._ut_op = FirstTimeDerivative(self.wavefield, self.time_order)
        self._utt_op = SecondTimeDerivative(self.wavefield, self.time_order)
        self._lap_op = Laplacian(self.wavefield, self.space_order)

        # precompute
        self.S = self.wavefield.grid.sig_mat  # damping coefficients
        c_sqr = sp.diags(self.model.c.ravel() ** 2)
        self.C2L = c_sqr @ self._lap_op.L

        self.nt = self.wavefield.grid.nt
        self.nx, self.ny = self.wavefield.grid.shape

    def matvec(self, u: np.ndarray) -> np.ndarray:
        """Compute the matrix vector product Au (no explicit A formation)"""
        U = u.reshape(self.nt, self.nx * self.ny)

        utt = self._utt_op.apply(U)
        damp = self._ut_op.apply(
            U @ self.S.T
        )  # damping is only applied in the boundary region
        lap = U @ self.C2L.T

        AU = utt + damp - lap

        # enforce ICs
        # IC1: u(0) = 0
        AU[0, :] = U[0, :]

        # IC2: ut(0) = 0
        AU[1, :] = (self._ut_op.Dt @ U)[1, :]
        return AU.ravel()

    def residual(self, u: np.ndarray, f: np.ndarray) -> np.ndarray:
        """Compute Au - f, where A encodes the derivatives and PML condition

        Note that sources may be a (n_txy * n_shots) matrix encoding each of the shots
        """
        return self.matvec(u) - f

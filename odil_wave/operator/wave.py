from dataclasses import dataclass, field
from typing import List, Tuple

from odil_wave.wavefield import Wavefield
from .temporal import SecondTimeDerivative
from .spatial import Laplacian
from odil_wave.models.base import VelocityModel
from .boundaries import HigdonBC

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

    _bcs: List[HigdonBC] = field(init=False)  # Higdon ABC

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

        # create BC objects for each boundary
        self._bcs = []
        bcs = ("left", "right", "top", "bottom")
        for b in bcs:
            bc = HigdonBC(
                self.wavefield, self.model, self.space_order, self.time_order, b
            )
            self._bcs.append(bc)

    def _apply_interior(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the action of the discrete operator A on a LHS vector u"""
        U = u.reshape(self.nt, self.nx * self.ny)

        utt = self._utt_op.apply(U)
        lap = U @ self.C2L.T

        # apply dt**2 scaling
        AU = self.dt2 * (utt - lap)
        return AU, U

    def _apply_bcs(self, AU: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Apply 2nd order Higdon ABCs to matrix-vector product AU"""
        # apply Higdon ABCs
        for bc in self._bcs:
            AU[:, bc.bdry_cols] = bc.apply(U)
        return AU

    def _apply_ic(self, AU: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Apply velocity and amplitude ICs U_t(0) = U(0) = 0"""
        # IC1: u(0) = 0
        AU[0, :] = U[0, :]

        # IC2: ut(0) = 0 - U[1] = U[0]  (1st-order forward diff from t=0)
        AU[1, :] = U[1, :]
        return AU.ravel()

    def apply_pde(self, u: np.ndarray) -> np.ndarray:
        """Apply the operator A to a vector u and Higdon BCs. This is the
        Toeplitz form of the product (no ICs, required separately for precond)
        """
        AU, U = self._apply_interior(u)
        return self._apply_bcs(AU, U)

    def matvec(self, u: np.ndarray) -> np.ndarray:
        """Compute the matrix vector product Au with IC and BC application"""
        AU, U = self.apply_pde(u)
        return self._apply_ic(AU, U)

    def rmatvec(self, r: np.ndarray) -> np.ndarray:
        """Compute the transposed matrix vector product A^T r
        (no explicit A formation)"""
        R = r.reshape(self.nt, self.nx * self.ny)

        Rz = R.copy()
        Rz[0, :] = 0.0  # adjoint of overwriting output rows 0,1:
        Rz[1, :] = 0.0  # the PDE terms must not see R[0], R[1]

        # zero boundary columns
        for bc in self._bcs:
            Rz[:, bc.bdry_cols] = 0.0

        utt_t = self._utt_op.apply_transpose(Rz)  # Dtt.T @ R

        lap_t = Rz @ self.C2L

        ATv = self.dt2 * (utt_t - lap_t)

        # transpose of IC constraints
        ATv[0, :] += R[0, :]
        ATv[1, :] += R[1, :]

        # zero IC rows before calling apply_transpose to avoid spurious contributions
        R_higdon = R.copy()
        R_higdon[0, :] = 0.0
        R_higdon[1, :] = 0.0

        for bc in self._bcs:
            ATv += bc.apply_transpose(R_higdon)

        return ATv.ravel()

    def residual(self, u: np.ndarray, f: np.ndarray) -> np.ndarray:
        """Compute Au - f, where A encodes the derivatives and boundary conditions

        Note that sources may be a (n_txy * n_shots) matrix encoding each of the shots
        """
        # apply dt**2 scaling to source term
        F = f.reshape(self.nt, self.nx * self.ny)
        Ff = self.dt2 * F
        Ff[0, :] = F[0, :]
        Ff[1, :] = F[1, :]
        return self.matvec(u) - Ff.ravel()

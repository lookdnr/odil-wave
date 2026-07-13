from dataclasses import dataclass, field
from typing import List, Tuple
from functools import cached_property

from odil_wave.wavefield import Wavefield
from .temporal import SecondTimeDerivative
from .spatial import Laplacian
from odil_wave.models.base import VelocityModel
from .boundaries import HigdonBC

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, factorized


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
        self.dt = self.wavefield.grid.dt
        self.dt2 = self.dt**2

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

    def apply_pde(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the operator A to a vector u and Higdon BCs. This is the
        Toeplitz form of the product (no ICs, required separately for precond)
        """
        AU, U = self._apply_interior(u)
        return self._apply_bcs(AU, U), U

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

    @cached_property
    def reduced_blocks(self) -> Tuple[sp.csr_array, ...]:
        """Build the time stencil blocks of the reduced system:

            B0 u_{m+1} + B1 u_m + B2 u_{m-1} = dt^2 f_m

        B0, B1, and B2 make up the time stencil, they are the block stencil
        coefficients where

            - B0 is whatever multiplies the next slice u_{m+1}
            - B1 is whatever multiplies the current slice u_{m}
            - B2 is whatever multiplies the previous slice u_{m-1}

        Building these separately is required for the downstream preconditioner
        since they allow us to build the Block Toeplitz with Toeplitz Blocks
        (BTTB) system

        TODO: currently hardcoded for 2nd order in time, generalising is a bigger task
        """
        if self.time_order > 2:
            raise NotImplementedError(
                "reduced blocks with time order > 2 are non-square."
            )

        ns, dt = self.nx * self.ny, self.dt
        ident = sp.identity(ns, format="csr")  # nxny, nxny identity

        # create a mask for BC application
        mask = np.ones(ns)
        for bc in self._bcs:
            mask[bc.bdry_cols] = 0.0

        # normal derivative weighted by local wavespeed
        # combine into (ns, ns) sparse matrix using elementwise sum()
        Dn = sum(
            bc.sign
            * (
                ident[bc.bdry_cols].T  # type: ignore direction * scatter (ns, n_bdry)
                @ sp.diags(bc.c_bdry)  # local c weights
                @ bc.Dn
            )  # normal derivative
            for bc in self._bcs
        )

        # normal second derivative weighted by c^2
        Dnn = sum(
            (
                ident[bc.bdry_cols].T  # type: ignore scatter (ns, n_bdry)
                @ sp.diags(bc.c_bdry**2)  # local c weights, squared
                @ bc.Dnn
            )  # second normal derivative
            for bc in self._bcs
        )

        # build blocks
        # previous term (1 in the time stencil - I) plus higdon
        B0 = (ident + dt * Dn).tocsr()

        # current term (-2 in time stencil) plus Laplacian term masked at boundaries
        # for BC application
        B1 = (
            -2 * ident - self.dt2 * (sp.diags(mask) @ self.C2L) + self.dt2 * Dnn
        ).tocsr()

        # next term (1 in time stencil) plus higdon
        B2 = (ident - dt * Dn).tocsr()
        return B0, B1, B2

    def reduced_operator(self) -> LinearOperator:
        """Return the BTTB operator for unknowns u_2 ... u_{nt-1}
        IC rows are clipped since they break the BTTB structure.

        Returned as scipy.sparse.linalg.LinearOperator for application in GMRES
        """
        ntm2, ns = self.nt - 2, self.nx * self.ny

        def matvec(u: np.ndarray) -> np.ndarray:
            """Matirx-vector product AU for the reduced system (IC rows clipped)"""
            U = np.zeros((self.nt, ns))
            U[2:] = u.reshape(ntm2, ns)  # clip to (nt-2, ns)
            AU, U = self.apply_pde(U.ravel())  # apply PDE
            return AU[1 : self.nt - 1].ravel()

        return LinearOperator(
            shape=(ntm2 * ns, ntm2 * ns),
            matvec=matvec,  # type: ignore
            dtype=np.float64,
        )

    def reduced_rhs(self, f: np.ndarray) -> Tuple[np.ndarray, ...]:
        """RHS of Au = f woth known IC slices eliminated from the RHS"""
        F = f.reshape(self.nt, self.nx * self.ny)
        f0, f1 = F[0], F[1]  # extract terms corresponding to IC

        # get reduced block operators
        _, B1, B2 = self.reduced_blocks

        # transfer known IC values to dt2 scaled rhs
        rhs = (self.dt2 * F[1 : self.nt - 1]).copy()

        # B0 u_{m-1} + B1 u_m + B2 u_{m+1} = dt2 f_m
        # put unknowns (beyond u_1) on LHS and knowns on RHS:
        # m = 1: B0 u_2 = dt2 f1 - B1 u_1 - B2 u_0
        # m = 2: B0 u_3  + B2 u_1= dt2 f2 - B1 u_2
        rhs[0] -= B1 @ f1 + B2 @ f0
        rhs[1] -= B2 @ f1
        return rhs.ravel(), f0, f1

    def march(self, f: np.ndarray) -> np.ndarray:
        """Solve the reduced system exactly by forward substitution in time.
        Useful for generating reference solutions.
        """
        B0, B1, B2 = self.reduced_blocks
        ns = self.nx * self.ny
        F = f.reshape(self.nt, ns)
        solve = factorized(B0.tocsc())  # factorise once and reuse

        U = np.empty((self.nt, ns))
        U[0], U[1] = F[0], F[1]  # ICs, same convention as reduced_rhs
        for m in range(1, self.nt - 1):
            #  B0 u_{m+1} + B1 u_m + B2 u_{m-1} = dt^2 f_m
            U[m + 1] = solve(self.dt2 * F[m] - B1 @ U[m] - B2 @ U[m - 1])
        return U

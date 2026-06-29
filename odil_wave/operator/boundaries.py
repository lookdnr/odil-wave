from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp

from odil_wave.wavefield import Wavefield
from odil_wave.models.base import VelocityModel
from .spatial import Laplacian
from .stencils import STENCIL_COEFFS_1ST, STENCIL_OFFSETS
from .ghost import GhostFill
from .temporal import FirstTimeDerivative, SecondTimeDerivative


def _first_diff_spatial(ord: int, n: int, h: float, ghost_width: int) -> sp.csr_matrix:
    """First x/y derivative, rectangular (n, n+2g) with ghost support"""

    top_coeffs, bot_coeff = STENCIL_COEFFS_1ST[ord]
    offsets = STENCIL_OFFSETS[ord]

    return sp.diags(
        np.array(top_coeffs),
        [o + ghost_width for o in offsets],  # type: ignore
        shape=(n, n + 2 * ghost_width),
        format="csr",
    ) / (bot_coeff * h)


@dataclass
class HigdonBC:
    """Higdon 2nd order ABC rows for one boundary edge.

    Enforces: u_tt + sign * 2 * u_nt + c_bdry^2 * u_nn = 0
    where n is the outward normal direction.
    """

    wavefield: Wavefield
    model: VelocityModel
    space_order: int
    time_order: int
    boundary: str  # "left" | "right" | "bottom" | "top"

    bdry_cols: np.ndarray = field(init=False)
    c_bdry: np.ndarray = field(init=False)
    sign: float = field(init=False)
    dt2: float = field(init=False)

    lap: Laplacian = field(init=False)

    Dtt: sp.csr_matrix = field(init=False)
    Dt: sp.csr_matrix = field(init=False)

    Dn: sp.csr_matrix = field(init=False)  # (n_bdry, nx*ny) first normal deriv
    Dnn: sp.csr_matrix = field(init=False)  # (n_bdry, nx*ny) second normal deriv

    def __post_init__(self):
        grid = self.wavefield.grid
        nx, ny = grid.nx, grid.ny
        g = self.space_order // 2

        self.Dtt = SecondTimeDerivative(self.wavefield, self.time_order).Dtt
        self.Dt = FirstTimeDerivative(self.wavefield, self.time_order).Dt
        self.dt2 = grid.dt**2

        self.lap = Laplacian(self.wavefield, self.space_order)

        if self.boundary in ("left", "right"):
            Dx = _first_diff_spatial(self.space_order, nx, grid.dx, g)  # (nx, nx+2g)
            Dxx = self.lap.Dxx

            Gx = GhostFill(nx, g).expand_x(ny)  # ((nx+2g)*ny, nx*ny)

            Dx_full = sp.csr_matrix(sp.kron(Dx, sp.eye(ny)) @ Gx)  # (nx*ny, nx*ny)
            Dxx_full = sp.csr_matrix(sp.kron(Dxx, sp.eye(ny)) @ Gx)

            if self.boundary == "left":
                self.bdry_cols = np.arange(ny)  # i=0, all j
                self.sign = -1.0  # left outward normal is -x
                self.Dn = Dx_full[:ny, :]  # 1st normal derivative
                self.Dnn = Dxx_full[:ny, :]  # 2nd normal derivative
            else:
                self.bdry_cols = np.arange((nx - 1) * ny, nx * ny)  # i=nx-1
                self.sign = 1.0  # right normal is +x
                self.Dn = Dx_full[(nx - 1) * ny :, :]
                self.Dnn = Dxx_full[(nx - 1) * ny :, :]

        else:  # "bottom" | "top"
            Dy = _first_diff_spatial(self.space_order, ny, grid.dy, g)
            Dyy = self.lap.Dyy

            Gy = GhostFill(ny, g).expand_y(nx)

            Dy_full = sp.csr_matrix(sp.kron(sp.eye(nx), Dy) @ Gy)
            Dyy_full = sp.csr_matrix(sp.kron(sp.eye(nx), Dyy) @ Gy)

            if self.boundary == "bottom":
                self.bdry_cols = np.arange(0, nx * ny, ny)  # j=0, all i
                self.sign = -1.0
                self.Dn = Dy_full[self.bdry_cols, :]
                self.Dnn = Dyy_full[self.bdry_cols, :]
            else:
                self.bdry_cols = np.arange(ny - 1, nx * ny, ny)  # j=ny-1
                self.sign = 1.0
                self.Dn = Dy_full[self.bdry_cols, :]
                self.Dnn = Dyy_full[self.bdry_cols, :]

        # extract local wavespeeds at boundary nodes
        self.c_bdry = self.model.c.ravel()[self.bdry_cols]  # (n_bdry,)

    def apply(self, U: np.ndarray) -> np.ndarray:
        """Higdon residual at this boundary. U: (nt, nx*ny) -> (nt, n_bdry)"""

        utt = self.Dtt @ U[:, self.bdry_cols]  # (nt, n_bdry)
        unn = U @ self.Dnn.T  # (nt, n_bdry)
        unt = self.Dt @ (U @ self.Dn.T)  # (nt, n_bdry)

        # u_tt + sign * 2 * u_nt + c_bdry^2 * u_nn
        # scaled by dt**2 for consistency
        return self.dt2 * (
            utt
            + self.sign * 2.0 * unt * self.c_bdry  # broadcast over time axis
            + unn * self.c_bdry**2
        )

    def apply_transpose(self, R: np.ndarray) -> np.ndarray:
        """Add Higdon adjoint to ATv. R: is the (nt, nx*ny) full residual."""

        Rb = R[:, self.bdry_cols]  # boundary residual (nt, n_bdry)

        ATv = np.zeros_like(R)
        ATv[:, self.bdry_cols] += self.dt2 * (self.Dtt.T @ Rb)
        ATv += self.dt2 * self.sign * 2.0 * (self.Dt.T @ (Rb * self.c_bdry)) @ self.Dn
        ATv += self.dt2 * (Rb * self.c_bdry**2) @ self.Dnn

        return ATv

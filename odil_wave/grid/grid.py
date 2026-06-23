import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
import scipy.sparse as sp


@dataclass
class Grid:
    """2D space + time discretization with a PML sponge layer wrapping the interior.

    The user specifies the *interior* (non-PML) shape and physical extent.
    The PML adds `pml_width` extra cells per side, extending the total grid
    (and total physical box) by `pml_width * dx` per side. The wavefield
    optimization variable lives on the full extended grid `(NT, NX, NY)`.
    """

    interior_shape: Tuple[int, int] = (100, 100)
    interior_extent: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (-1.0, 1.0),
        (-1.0, 1.0),
    )

    # grid layout
    interior_nx: int = field(init=False)
    interior_ny: int = field(init=False)

    nx: int = field(init=False)
    ny: int = field(init=False)
    nt: int = field(init=False)

    dx: float = field(init=False)
    dy: float = field(init=False)
    dt: float = field(init=False)

    t_max: Optional[float] = None  # total simulation duration
    init_nt: Optional[int] = None  # optional override; derived from CFL if None

    extent: Tuple[Tuple[float, float], Tuple[float, float]] = field(init=False)

    # coordinate grids (for usage elsewhere)
    x: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)
    t: np.ndarray = field(init=False, repr=False)
    X: np.ndarray = field(init=False, repr=False)
    Y: np.ndarray = field(init=False, repr=False)

    # PML damping coefficients in each direction
    sigma_x: np.ndarray = field(init=False, repr=False)
    sigma_y: np.ndarray = field(init=False, repr=False)

    c_ref: float = 1.5  # reference wavespeed
    cfl_safety: float = 0.8  # fraction of theoretical safety to use for dt
    pml_width: int = 10  # extra cells per side wrapping the interior
    pml_power: int = 3  # sigma(d) = sigma_max * (d / L_pml)^pml_power
    pml_R0: float = 1e-6  # target theoretical reflection coefficient

    def __post_init__(self):
        self.interior_nx, self.interior_ny = self.interior_shape
        (ix_min, ix_max), (iy_min, iy_max) = self.interior_extent

        self.dx = (ix_max - ix_min) / (self.interior_nx - 1)
        self.dy = (iy_max - iy_min) / (self.interior_ny - 1)

        p = self.pml_width
        self.nx = self.interior_nx + 2 * p
        self.ny = self.interior_ny + 2 * p
        x_min = ix_min - p * self.dx
        x_max = ix_max + p * self.dx
        y_min = iy_min - p * self.dy
        y_max = iy_max + p * self.dy
        self.extent = ((x_min, x_max), (y_min, y_max))

        # TODO: check proper way of setting t_max and nt
        if self.t_max is None:
            diag = math.hypot(ix_max - ix_min, iy_max - iy_min)
            self.t_max = 2.0 * diag / self.c_ref
        if self.init_nt is None:
            dt_cfl = 1.0 / (self.c_ref * math.sqrt(1.0 / self.dx**2 + 1.0 / self.dy**2))
            self.nt = int(math.ceil(self.t_max / (self.cfl_safety * dt_cfl))) + 1
        else:
            self.nt = self.init_nt
        self.dt = self.t_max / (self.nt - 1)

        self.x = np.linspace(x_min, x_max, self.nx)
        self.y = np.linspace(
            y_min,
            y_max,
            self.ny,
        )
        self.t = np.linspace(
            0.0,
            self.t_max,
            self.nt,
        )
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        self.sigma_x, self.sigma_y = self._build_pml_profiles()
        self.sig_mat = self._sigma_matrix()

    @property
    def shape(self) -> Tuple[int, int]:
        """Total grid shape (interior + PML)."""
        return (self.nx, self.ny)

    @property
    def interior_slice(self) -> Tuple[slice, slice]:
        """Slice into a full-grid ndarray that picks out the interior."""
        p = self.pml_width
        return (slice(p, p + self.interior_nx), slice(p, p + self.interior_ny))

    def _sigma_max(self, L_pml_phys: float) -> float:
        """sigma_max from a target theoretical reflection coefficient."""
        if L_pml_phys <= 0:
            raise ValueError("L_pml_phys must be positive.")
        if self.c_ref <= 0:
            raise ValueError("c_ref must be positive.")
        if not (0.0 < self.pml_R0 < 1.0):
            raise ValueError("pml_R0 must lie strictly between 0 and 1.")

        return -((self.pml_power + 1) * self.c_ref * math.log(self.pml_R0)) / (
            2.0 * L_pml_phys
        )

    def _build_pml_profiles(self):
        """sigma_x(i, j), sigma_y(i, j) on the full grid; zero in the interior."""
        p = self.pml_width
        if p == 0:
            zeros = np.zeros((self.nx, self.ny))
            return zeros, zeros
        L_pml_x = p * self.dx
        L_pml_y = p * self.dy
        sigma_max_x = self._sigma_max(L_pml_x)
        sigma_max_y = self._sigma_max(L_pml_y)

        i = np.arange(self.nx)
        j = np.arange(self.ny)

        d_x = (
            np.maximum(p - i, 0.0) + np.maximum(i - (self.nx - 1 - p), 0.0)
        ) * self.dx
        d_y = (
            np.maximum(p - j, 0.0) + np.maximum(j - (self.ny - 1 - p), 0.0)
        ) * self.dy

        sigma_x_1d = sigma_max_x * (d_x / L_pml_x) ** self.pml_power
        sigma_y_1d = sigma_max_y * (d_y / L_pml_y) ** self.pml_power

        sigma_x = np.broadcast_to(sigma_x_1d.reshape(-1, 1), (self.nx, self.ny)).copy()
        sigma_y = np.broadcast_to(sigma_y_1d.reshape(1, -1), (self.nx, self.ny)).copy()

        return sigma_x, sigma_y

    def _sigma_matrix(self):
        """Compute the diagonal damping matrix for enforcing the PML"""
        sigma = self.sigma_x + self.sigma_y
        return sp.diags(sigma.ravel())

    def cfl(self, c_max: float) -> float:
        return c_max * self.dt * math.sqrt(1.0 / self.dx**2 + 1.0 / self.dy**2)

    def plot_absorption_profile(self, ax=None):
        """Plot the 2D PML absorption (sigma_x + sigma_y) over the full grid."""
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 4.5))
        (xmin, xmax), (ymin, ymax) = self.extent
        im = ax.imshow(
            (self.sigma_x + self.sigma_y),
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap="magma",
            aspect="equal",
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(r"PML absorption $\sigma_x + \sigma_y$")
        (ix0, ix1), (iy0, iy1) = self.interior_extent
        ax.add_patch(
            Rectangle(
                (ix0, iy0),
                ix1 - ix0,
                iy1 - iy0,
                fill=False,
                edgecolor="white",
                linestyle="--",
                linewidth=1.0,
                label="non-PML interior",
            )
        )
        plt.colorbar(im, ax=ax, shrink=0.85, label=r"$\sigma$ [1/s]")
        return ax

    @property
    def summary(self) -> str:
        (ix0, ix1), (iy0, iy1) = self.interior_extent
        (xmin, xmax), (ymin, ymax) = self.extent
        return (
            f"Grid interior {self.interior_nx}x{self.interior_ny} -> "
            f"total {self.nx}x{self.ny} (PML p={self.pml_width}),"
            f"\nnt={self.nt}, dx={self.dx:.4f}m, dy={self.dy:.4f}m,"
            f"\ndt={self.dt:.4f} s, cfl@c_ref={self.cfl(self.c_ref):.3f},"
            f"\ninterior x in [{ix0:.2f}, {ix1:.2f}]m,"
            f"\ny in [{iy0:.2f}, {iy1:.2f}]m,"
            f"\ntotal x in [{xmin:.2f}, {xmax:.2f}]m,"
            f"\ntotal y in [{ymin:.2f}, {ymax:.2f}]m"
        )

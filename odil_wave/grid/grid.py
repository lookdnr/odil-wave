import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class Grid:
    """2D space + time discretization grid.

    The user specifies the (nx, ny) shape and physical extent.
    The wavefield optimisation variable lives on the full extended grid `(NT, NX*NY)`.
    """

    # spatial parameters
    xmin: float = 0.0
    xmax: float = 1.0
    ymin: float = 0.0
    ymax: float = 1.0
    t_max: Optional[float] = None  # total simulation duration

    # grid layout
    nx: int = 100
    ny: int = 100
    nt: int = field(init=False)

    # grid spacings
    dx: float = field(init=False)
    dy: float = field(init=False)
    dt: float = field(init=False)

    _extent: Tuple[Tuple[float, float], Tuple[float, float]] = field(init=False)

    # coordinate grids (for usage elsewhere)
    x: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)
    t: np.ndarray = field(init=False, repr=False)
    X: np.ndarray = field(init=False, repr=False)
    Y: np.ndarray = field(init=False, repr=False)

    c_ref: float = 1.5  # reference wavespeed
    cfl_safety: float = 0.8  # fraction of theoretical safety to use for dt

    def __post_init__(self):

        # ===== catch extent errors =====
        if self.xmax == self.xmin:  # catch equal xmax, xmin
            raise ValueError("args xmax and xmin cannot be equal.")
        elif self.xmax < self.xmin:  # swap xmax and xmin if neccesary
            temp = self.xmin
            self.xmin = self.xmax  # xmin becomes xmax
            self.xmax = temp  # xmax becomes xmin

        if self.ymax == self.ymin:  # catch equal xmax, xmin
            raise ValueError("args xmax and xmin cannot be equal.")
        elif self.ymax < self.ymin:  # swap xmax and xmin if neccesary
            temp = self.ymin
            self.ymin = self.ymax  # xmin becomes xmax
            self.ymax = temp  # xmax becomes xmin

        # ===== catch nx, nx errors =====

        if (self.nx < 3) or (self.ny < 3):  # 3 for central diff
            raise ValueError(
                f"args nx, ny cannot be less than 3, got nx = {self.nx}, ny = {self.ny}"
            )

        # ===== catch c_cfl, c_ref errors =====
        if self.c_ref <= 0:
            raise ValueError(
                f"arg c_ref should be strictly positive, got c_ref = {self.c_ref}"
            )

        if not (0 < self.cfl_safety < 1.0):
            raise ValueError(
                f"arg cfl_safety must be less than 1, got {self.cfl_safety}"
            )

        # compute grid spacing
        self.dx = (self.xmax - self.xmin) / (self.nx - 1)
        self.dy = (self.ymax - self.ymin) / (self.ny - 1)

        # compute spatial extent
        self._extent = ((self.xmin, self.xmax), (self.ymin, self.ymax))

        # compute duration as time for wave to cross domain at c_ref
        if self.t_max is None:
            diag = math.hypot(self.xmax - self.xmin, self.ymax - self.ymin)
            self.t_max = 2.0 * diag / self.c_ref

        dt_cfl = 1.0 / (self.c_ref * math.sqrt(1.0 / self.dx**2 + 1.0 / self.dy**2))
        self.nt = int(math.ceil(self.t_max / (self.cfl_safety * dt_cfl))) + 1

        self.dt = self.t_max / (self.nt - 1)

        # compute meshes for later use
        self.x = np.linspace(self.xmin, self.xmax, self.nx)
        self.y = np.linspace(
            self.ymin,
            self.ymax,
            self.ny,
        )
        self.t = np.linspace(
            0.0,
            self.t_max,
            self.nt,
        )
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

    @property
    def shape(self) -> Tuple[int, int]:
        """Spatial grid shape (nx, ny)."""
        return (self.nx, self.ny)

    @property
    def extent(self) -> Tuple[Tuple[float, float], ...]:
        """Get the spatial extent of the grid"""
        return self._extent

    def cfl(self, c_max: float) -> float:
        """Compute the value of the CFL condition for some wavespee c_max"""
        return c_max * self.dt * math.sqrt(1.0 / self.dx**2 + 1.0 / self.dy**2)

    @property
    def summary(self) -> str:
        (xmin, xmax), (ymin, ymax) = self.extent
        return (
            f"Nx, Ny, Nt: {self.nx}, {self.ny}, {self.nt}"
            f"\ndx, dy, dt: {self.dx:.4f}m, {self.dy:.4f}m, {self.dt:.4f}s"
            f"\n,CFL (at c = {self.c_ref}):  {self.cfl(self.c_ref):.3f}"
            f"\nx in [{xmin:.2f}, {xmax:.2f}]m"
            f"\ny in [{ymin:.2f}, {ymax:.2f}]m"
        )

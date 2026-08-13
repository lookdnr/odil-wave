import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .utils import points_per_wavelength, nodes_for_ppw

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

    # speed
    c_min: float = 1.5  # reference wavespeed
    c_max: float = 2.0
    cfl_safety: float = 0.8  # fraction of theoretical safety to use for dt
    allow_unstable: bool = False  # allow unstable cfl condtions

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

        # ===== catch c_cfl, c errors =====
        if self.c_min <= 0 or self.c_max <= 0:
            raise ValueError(
                "args c_min, c_max should be strictly positive,"
                + f" got {self.c_min, self.c_max}"
            )
        elif self.c_min > self.c_max:
            temp = self.c_min
            self.c_min = self.c_max
            self.c_max = temp

        if self.cfl_safety <= 0:
            raise ValueError(f"arg cfl_safety must be positive , got {self.cfl_safety}")
        if self.cfl_safety >= 1.0 and not self.allow_unstable:
            raise ValueError(
                f"arg cfl_safety must be < 1, got {self.cfl_safety}. "
                "pass allow_unstable=True to intentionally exceed the CFL limit"
            )

        # compute grid spacing
        self.dx = (self.xmax - self.xmin) / (self.nx - 1)
        self.dy = (self.ymax - self.ymin) / (self.ny - 1)

        # compute spatial extent
        self._extent = ((self.xmin, self.xmax), (self.ymin, self.ymax))

        # compute duration as time for wave to cross domain at c_min
        if self.t_max is None:
            diag = math.hypot(self.xmax - self.xmin, self.ymax - self.ymin)
            self.t_max = 1.2 * diag / self.c_min

        dt_cfl = 1.0 / (self.c_max * math.sqrt(1.0 / self.dx**2 + 1.0 / self.dy**2))
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

    def ppw(self, f0: float, c: Optional[float] = None) -> float:
        """Compute points per shortest wavelength (lambda = c / f0)"""
        c = self.c_min if c is None else c
        return points_per_wavelength(max(self.dx, self.dy), f0, c)

    @classmethod
    def from_ppw(
        cls,
        f0: float,
        ppw: float,
        *,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        c_min: float,
        c_max: float,
        cfl_safety: float = 0.8,
        t_max: Optional[float] = None,
    ) -> "Grid":
        """Construct a Grid sized to resolve at least `ppw` points per shortest
        wavelength (lambda = c_min / f0) on each axis."""
        nx = nodes_for_ppw(xmax - xmin, f0, ppw, c_min)
        ny = nodes_for_ppw(ymax - ymin, f0, ppw, c_min)
        return cls(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            nx=nx,
            ny=ny,
            c_min=c_min,
            c_max=c_max,
            cfl_safety=cfl_safety,
            t_max=t_max,
        )

    @property
    def summary(self) -> str:
        (xmin, xmax), (ymin, ymax) = self.extent
        return (
            f"Nx, Ny, Nt: {self.nx}, {self.ny}, {self.nt}"
            f"\ndx, dy, dt: {self.dx:.9f}m, {self.dy:.9f}m, {self.dt:.9f}s"
            f"\nCFL (at c_max = {self.c_max}):  {self.cfl(self.c_max):.3f}"
            f"\nx in [{xmin:.3f}, {xmax:.3f}]m"
            f"\ny in [{ymin:.3f}, {ymax:.3f}]m"
        )

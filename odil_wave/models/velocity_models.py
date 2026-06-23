from typing import Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from skimage.data import shepp_logan_phantom
from skimage.transform import resize

import numpy as np

from odil_wave.grid import Grid


@dataclass
class VelocityModel(ABC):
    """2D velocity field c(x, y) attached to a Grid (full extended grid)."""

    grid: Grid  # discrete grid
    background_c: float = 1.0  # background wave speed
    contrast: float = 0.7  # anomaly constrast vs background
    model: np.ndarray = field(init=False)  # data
    name: str = field(init=False)  # identifier

    def __post_init__(self):

        # warn about negative wavespeed
        if self.background_c < 0.0:
            warnings.warn(
                f"background_c = {self.background_c} is nonphysical;"
                + " wavespeed should be positive.",
                UserWarning,
            )

        # warn about negative contrast (creates negative wavespeed in model)
        if self.contrast < 0.0:
            warnings.warn(
                f"contrast = {self.contrast} will create a nonphysical model;"
                + "contrast should be positive for velocity models.",
                UserWarning,
            )

        self.name = "Base class"

    @abstractmethod
    def _build(self) -> np.ndarray:
        pass

    @property
    def c_max(self) -> float:
        return float(self.model.max())

    @property
    def c_min(self) -> float:
        return float(self.model.min())

    def show(
        self,
        ax=None,
        title: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show_pml: bool = True,
    ):
        # create ax if not specified
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 4.5))

        (xmin, xmax), (ymin, ymax) = self.grid.extent
        im = ax.imshow(
            self.model,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )

        # labels and colorbar
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        ax.set_title(title or f"c(x, y) [{self.name}]")

        plt.colorbar(im, ax=ax, shrink=0.85, label="c [m/s]")

        # add patch on plot indicating PML
        if show_pml:
            (ix0, ix1), (iy0, iy1) = self.grid.interior_extent
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

        return ax


class HomogeneousModel(VelocityModel):
    """Homeogenous velocity odel: background_c everywhere"""

    def __init__(self, grid: Grid, background_c: float):
        super().__init__(
            grid, background_c, contrast=1.0
        )  # no contrast for homeogenous model
        self.model = self._build()
        self.name = "Homogeneous Model"

    def _build(self):
        return np.full(self.grid.shape, self.background_c)


class SheppLoganModel(VelocityModel):
    """Shepp-Logan phantom velocity model"""

    def __init__(self, grid, base, contrast, interior_fill: float = 0.7):
        super().__init__(grid, base, contrast)
        self.interior_fill = (
            interior_fill  # fraction of the interior grid the model should fill
        )
        self.model = self._build()
        self.name = "Shepp-Logan Phantom Model"

    def _build(self):

        s_nx = max(2, int(self.grid.interior_nx * self.interior_fill))
        s_ny = max(2, int(self.grid.interior_ny * self.interior_fill))
        phantom = shepp_logan_phantom().astype(np.float32)

        # Rotate 90deg so the phantom's long axis aligns with the
        # AcquisitionGeometry ellipse's semi-major axis (y).
        phantom = np.rot90(phantom, k=1).copy()
        phantom = resize(phantom, (s_nx, s_ny), anti_aliasing=True, mode="reflect")

        # compute centre point
        p = self.grid.pml_width
        i0 = p + (self.grid.interior_nx - s_nx) // 2
        j0 = p + (self.grid.interior_ny - s_ny) // 2

        # create base and add anomalies
        c = np.full(self.grid.shape, self.background_c)
        c[i0 : i0 + s_nx, j0 : j0 + s_ny] = self.background_c + self.contrast * phantom
        return c


class OverDensityModel(VelocityModel):
    """Cicular anomaly model"""

    def __init__(
        self,
        grid: Grid,
        background_c: float,
        contrast: float = 0.7,
        centre: Tuple[float, float] = (0.0, 0.0),
        radius: float = 0.3,
    ) -> None:
        super().__init__(grid, background_c, contrast)
        self.name = "Circular Anomaly Model"

        # check radius is positive
        if radius <= 0.0:
            raise ValueError(f"arg `radius` must be > 0. Got {radius}.")

        self.radius = radius

        (x_min, x_max), (y_min, y_max) = self.grid.extent

        # check centre is within grid extent
        if not (x_min <= centre[0] <= x_max):
            raise ValueError(
                f"x-component of arg 'centre' must be between {x_min, x_max},"
                + " got {centre[0]}."
            )

        if not (y_min <= centre[0] <= y_max):
            raise ValueError(
                f"y-component of arg 'centre' must be between {y_min, y_max},"
                + " got {centre[1]}."
            )

        self.centre = centre

        self.model = self._build()

    def _build(self) -> np.ndarray:
        # create circular mask
        mask = (self.grid.X - self.centre[0]) ** 2 + (
            self.grid.Y - self.centre[1]
        ) ** 2 <= self.radius**2

        # create base model
        base = np.full(self.grid.shape, self.background_c)

        # apply mask to background with contrast
        return np.where(
            mask, np.full_like(base, self.background_c + self.contrast), base
        )

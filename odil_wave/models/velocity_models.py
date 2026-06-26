from typing import Tuple

from skimage.data import shepp_logan_phantom
from skimage.transform import resize

import numpy as np

from odil_wave.grid import Grid
from .base import VelocityModel


class HomogeneousModel(VelocityModel):
    """Homeogenous velocity odel: background_c everywhere"""

    def __init__(self, grid: Grid, background_c: float = 1.0):
        super().__init__(
            grid, background_c, contrast=1.0
        )  # no contrast for homeogenous model
        self.c = self._build()
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
        self.c = self._build()
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
        background_c: float = 1.0,
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
                + f" got {centre[0]}."
            )

        if not (y_min <= centre[1] <= y_max):
            raise ValueError(
                f"y-component of arg 'centre' must be between {y_min, y_max},"
                + " got {centre[1]}."
            )

        self.centre = centre

        self.c = self._build()

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

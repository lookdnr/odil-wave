from typing import Tuple

from skimage.data import shepp_logan_phantom
from skimage.transform import resize

import numpy as np

from odil_wave.grid import Grid
from .base import VelocityModel


class HomogeneousModel(VelocityModel):
    """Homogeneous velocity model: `background_c` everywhere.

    Parameters
    ----------
    grid : Grid
        Grid the velocity field is defined on.
    background_c : float, optional
        Uniform wavespeed across the domain.
    """

    def __init__(self, grid: Grid, background_c: float = 1.0):
        super().__init__(
            grid, background_c, contrast=1.0
        )  # no contrast for homeogenous model
        self.c = self._build()
        self.name = "Homogeneous Model"

    def _build(self):
        return np.full(self.grid.shape, self.background_c)


class SheppLoganModel(VelocityModel):
    """Shepp Logan phantom velocity model.

    Parameters
    ----------
    grid : Grid
        Grid the velocity field is defined on.
    background_c : float, optional
        Background wavespeed outside the phantom.
    contrast : float, optional
        Phantom intensity scale multiplier added to `background_c`.
    interior_fill : float, optional
        Fraction of the grid interior the phantom is resized to fill.
    mask_skull : bool, optional
        If True, threshold out the skull ring and fill it
        with the median interior brain intensity.
    centre_frac : tuple of (float, float), optional
        Placement of the phantom within its available margin, as a
        fraction (0=left/ bottom aligned, 1=right/top aligned) of the
        free space.
    """

    def __init__(
        self,
        grid,
        background_c: float = 1.0,
        contrast: float = 1.0,
        interior_fill: float = 0.7,
        mask_skull: bool = False,
        centre_frac: Tuple[float, float] = (0.5, 0.5),
    ):
        super().__init__(grid, background_c, contrast)
        self.interior_fill = (
            interior_fill  # fraction of the interior grid the model should fill
        )
        # we might want to mask the skull for forward modelling purposes
        self.mask_skull = mask_skull
        self.centre_frac = centre_frac
        self.c = self._build()
        self.name = "Shepp-Logan Phantom Model"

    def _build(self):

        s_nx = max(2, int(self.grid.nx * self.interior_fill))
        s_ny = max(2, int(self.grid.ny * self.interior_fill))
        phantom = shepp_logan_phantom().astype(np.float32)

        # Rotate 90deg so the phantom's long axis aligns with the
        # AcquisitionGeometry ellipse's semi-major axis (y).
        phantom = np.rot90(phantom, k=1).copy()

        if self.mask_skull:
            # remove skull by thresholding
            skull = phantom >= 0.9 * phantom.max()
            brain = phantom[(phantom > 0.0) & ~skull]
            fill = np.median(brain) if brain.size else 0.0
            phantom[skull] = fill

        phantom = resize(phantom, (s_nx, s_ny), anti_aliasing=True, mode="reflect")

        # position within available margin
        i0 = int(np.clip(self.centre_frac[0], 0.0, 1.0) * (self.grid.nx - s_nx))
        j0 = int(np.clip(self.centre_frac[1], 0.0, 1.0) * (self.grid.ny - s_ny))

        # create base and add anomalies
        c = np.full(self.grid.shape, self.background_c)
        c[i0 : i0 + s_nx, j0 : j0 + s_ny] = self.background_c + self.contrast * phantom
        return c


class OverDensityModel(VelocityModel):
    """Circular anomaly velocity model.

    Parameters
    ----------
    grid : Grid
        Grid the velocity field is defined on.
    background_c : float, optional
        Background wavespeed outside the anomaly.
    contrast : float, optional
        Wavespeed added to `background_c` inside the anomaly.
    centre : tuple of (float, float), optional
        Anomaly centre in spatial coordinates.
    radius : float, optional
        Anomaly radius.

    Raises
    ------
    ValueError
        If `radius` is not positive, or `centre` falls outside the grid extent.
    """

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


class CustomModel(VelocityModel):
    """Velocity model built from an arbitrary numpy array.

    The array is resized (with anti-aliasing) to the grid's (nx, ny)
    shape. `background_c` is set to the array's mean.

    Parameters
    ----------
    grid : Grid
        Grid the velocity field is defined on.
    c_array : np.ndarray
        Source wavespeed array, resized onto `grid`.
    """

    def __init__(self, grid: Grid, c_array: np.ndarray):
        super().__init__(grid, background_c=float(c_array.mean()), contrast=1.0)
        self._c_array = c_array
        self.c = self._build()
        self.name = "Array Model (from file)"

    def _build(self) -> np.ndarray:
        return np.array(
            resize(
                self._c_array,
                self.grid.shape,
                anti_aliasing=True,
                mode="reflect",
                preserve_range=True,
            )
        )

from typing import Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import warnings

import numpy as np
import matplotlib.pyplot as plt

from odil_wave.grid import Grid


@dataclass
class VelocityModel(ABC):
    """2D velocity field attached to a Grid.

    Subclasses implement `_build` to construct `c` from `background_c`/`contrast`
    and any model specific geometry.

    Parameters
    ----------
    grid : Grid
        Grid the velocity field is defined on.
    background_c : int or float, optional
        Background wavespeed.
    contrast : float, optional
        Anomaly contrast multiplier relative to the background.

    Attributes
    ----------
    c : np.ndarray
        (nx, ny) wavespeed field, built by `_build` in each subclass.
    name : str
        Model identifier, set by each subclass.

    Warns
    -----
    UserWarning
        If `background_c` or `contrast` is negative (nonphysical wavespeed).
    """

    grid: Grid  # discrete grid
    background_c: int | float = 1.0  # background wave speed
    contrast: float = 0.7  # anomaly constrast vs background
    c: np.ndarray = field(init=False)  # data
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
        if isinstance(self.background_c, int):
            self.background_c = float(self.background_c)

    @abstractmethod
    def _build(self) -> np.ndarray:
        """Construct and return the wavespeed model."""
        pass

    @property
    def c_max(self) -> float:
        """float: Maximum wavespeed in the field"""
        return float(self.c.max())

    @property
    def c_min(self) -> float:
        """float: Maximum wavespeed in the field"""
        return float(self.c.min())

    def show(
        self,
        ax=None,
        title: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "viridis",
    ):
        """Plot the velocity field c(x, y).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on, a new figure is created if None.
        title : str, optional
            Plot title, defaults to "c(x, y) [<model name>]".
        vmin, vmax : float, optional
            Colour scale limits passed to `imshow`.
        cmap : str, optional
            Colourmap name.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the field was plotted on.
        """
        # create ax if not specified
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 4.5))

        (xmin, xmax), (ymin, ymax) = self.grid.extent
        im = ax.imshow(
            self.c.T,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        # labels and colorbar
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        ax.set_title(title or f"c(x, y) [{self.name}]", pad=10)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"c ($ms^{-1}$)")

        return ax

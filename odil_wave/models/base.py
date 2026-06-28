from typing import Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from odil_wave.grid import Grid


@dataclass
class VelocityModel(ABC):
    """2D velocity field c(x, y) attached to a Grid (full extended grid)."""

    grid: Grid  # discrete grid
    background_c: float = 1.0  # background wave speed
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

    @abstractmethod
    def _build(self) -> np.ndarray:
        pass

    @property
    def c_max(self) -> float:
        return float(self.c.max())

    @property
    def c_min(self) -> float:
        return float(self.c.min())

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
            self.c.T,
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

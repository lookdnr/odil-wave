from typing import Optional, Tuple

import math
import numpy as np

import matplotlib.pyplot as plt

from odil_wave.grid import Grid
from odil_wave.models.base import VelocityModel

from .utils import place_ellipse, build_weight_matrix


class Sources:
    """Create an array of Ricker-in-time sources.

    If mode == "custom", custom placement (in spatial coordinates) can be used
    else if mode == "ring", sources will be placed in an ellipse controlled by
    the ring_centre and a_frac/ b_frac.

    Source injection is handled by sinc interpolation over an n_sinc window in
    each spatial direction.

    `n_receivers` transducer positions record every shot. A subset
    (`n_sources`, evenly spaced) act as shot sources.
    """

    def __init__(
        self,
        grid: Grid,
        n_sources: int | None = None,
        f0: float = 4.0,
        t0: Optional[float] = None,
        mode: str = "custom",  # default to custom placement
        source_locs: (
            Tuple[Tuple[float, float], ...] | None
        ) = None,  # spatial source locs
        a_frac: float = 0.55,
        b_frac: float = 0.70,
        ring_centre: Tuple[float, float] = (0.0, 0.0),
        n_sinc: int = 8,  # sinc window half-width (nodes per dimension)
    ):
        # inherit geometry from grid
        self.grid = grid

        if n_sources is None:
            n_sources = 1

        # validate source input
        if not n_sources > 0:
            raise ValueError(f"arg n_sources must be greater than 0, got {n_sources}")

        self.n_sources = n_sources

        # extract extent for validity checking
        (x_min, x_max), (y_min, y_max) = grid.extent

        # validate mode
        self.mode = mode.strip().lower()

        if self.mode not in ("custom", "ring"):
            raise ValueError(f"arg mode must be one of 'custom' 'ring', got {mode}")

        if self.mode == "ring":
            # validate a_frac, b_frac input
            if a_frac is None or a_frac <= 0:
                raise ValueError(f"arg a_frac should be > 0, got {a_frac}")

            if b_frac is None or b_frac <= 0:
                raise ValueError(f"arg b_frac should be > 0, got {b_frac}")

            self.a_frac = a_frac  # aspect ratio
            self.b_frac = b_frac  # aspect ratio

            x, y = ring_centre

            # validate ring_centre input
            if not (x_min < x < x_max):
                raise ValueError(
                    f"x component of arg ring_centre must be in range {x_min, x_max}"
                    f" for the given Grid, got {x}."
                )

            if not (y_min < y < y_max):
                raise ValueError(
                    f"y component of arg ring_centre must be in range {y_min, y_max}"
                    f" for the given Grid, got {y}."
                )

            self.ring_centre = ring_centre

            self.src_ij = place_ellipse(grid, n_sources, ring_centre, a_frac, b_frac)
            # physical coords for sinc injection
            self.src_xy = np.array(
                [[grid.x[i], grid.y[j]] for i, j in self.src_ij], dtype=float
            )

        if self.mode == "custom":
            if source_locs is None:
                raise ValueError(
                    "arg source_locs cannot be None with mode == 'custom'."
                )

            # validate source locs
            for k, (x, y) in enumerate(source_locs):
                if not (x_min < x < x_max):
                    raise ValueError(
                        f"x component of source_locs must be in range {x_min, x_max}"
                        f" for the given Grid, got {x} for source location {k}."
                    )

                if not (y_min < y < y_max):
                    raise ValueError(
                        f"y component of source_locs must be in range {y_min, y_max}"
                        f" for the given Grid, got {y} for source location {k}."
                    )

            self.src_xy = np.array(source_locs, dtype=float)  # (n_sources, 2)
            # snap to nearest node for display / indexing
            i = (
                np.round((self.src_xy[:, 0] - x_min) / grid.dx)
                .clip(0, grid.nx - 1)
                .astype(int)
            )
            j = (
                np.round((self.src_xy[:, 1] - y_min) / grid.dy)
                .clip(0, grid.ny - 1)
                .astype(int)
            )
            self.src_ij = np.stack([i, j], axis=-1)

        if f0 < 0:
            raise ValueError(f"arg f0 must be greater than 0, got {f0}.")

        # source config
        self.f0 = f0  # peak frequency
        self.t0 = 1.0 / f0 if t0 is None else t0  # causal Ricker delay [s]

        self.W = build_weight_matrix(
            grid, self.src_xy, self.n_sources, n_sinc
        )  # (nx*ny, n_sources)

    def ricker(self, t: np.ndarray) -> np.ndarray:
        arg = (math.pi * self.f0 * (t - self.t0)) ** 2
        return (1.0 - 2.0 * arg) * np.exp(-arg)

    def src_position(self, src_idx: int) -> Tuple[float, float]:
        return float(self.src_xy[src_idx, 0]), float(self.src_xy[src_idx, 1])

    def source_field(self, src_idx: int) -> np.ndarray:
        """(NT, NX, NY) sinc-injected point source field for shot src_idx."""
        w = self.W[:, src_idx].reshape(self.grid.shape)  # (nx, ny)
        temporal = self.ricker(self.grid.t)  # (nt,)
        # broadcast (nt, 1, 1) * (1, nx, ny) => (nt, nx, ny)
        return temporal.reshape(-1, 1, 1) * w.reshape(1, *self.grid.shape)

    def source_matrix(self) -> np.ndarray:
        """Precompute (nt*nx*ny, n_sources) source matrix for all shots."""
        f = self.ricker(self.grid.t)  # (nt,)

        # einsum multiples the source f and sinc weights matrix W and creates a new axis
        # f: (nt,), W: (nx*ny, n_sources)
        # output: (nt, nx*ny) -> (nt*ny*ny, n_sources) (expected format for solve)
        return np.einsum("t,ns->tns", f, self.W).reshape(-1, self.n_sources)

    def plot_source_pulse(self, ax=None):
        """Plot the Ricker temporal waveform R(t) of the source."""
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 3.5))
        r = self.ricker(self.grid.t)
        peak_t = int(np.argmax(np.abs(r)))
        ax.plot(self.grid.t, r)
        ax.axvline(self.grid.t[peak_t], color="gray", ls=":", alpha=0.7)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("R(t) [a.u.]")
        ax.set_title(rf"Ricker pulse ($f_0$={self.f0} Hz, $t_0$={self.t0:.3f} s)")
        ax.grid(alpha=0.3)
        return ax

    def plot_source_field(self, src_idx: int = 0, t_idx: Optional[int] = None, ax=None):
        """Plot a spatial snapshot of the source field s(x, y, t_idx).

        Defaults to the peak time of the Ricker pulse.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 4.5))
        src = self.source_field(src_idx)
        if t_idx is None:
            t_idx = int(np.argmax(np.abs(self.ricker(self.grid.t))))
        (xmin, xmax), (ymin, ymax) = self.grid.extent
        vmax = float(np.max(np.abs(src))) * 1.05 + 1e-12
        im = ax.imshow(
            src[t_idx].T,
            origin="lower",
            aspect="equal",
            extent=(xmin, xmax, ymin, ymax),
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"source field s(x, y, t={self.grid.t[t_idx]:.3f} s)")
        plt.colorbar(im, ax=ax, shrink=0.85, label="amplitude [a.u.]")
        return ax

    def show(self, velocity_model: VelocityModel, ax=None):
        """Plot the source geometry"""
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 5))
        velocity_model.show(
            ax=ax, title=f"Sources on {velocity_model.name}", show_pml=True
        )

        sx = self.grid.x[self.src_ij[:, 0]]
        sy = self.grid.y[self.src_ij[:, 1]]

        ax.scatter(
            sx,
            sy,
            marker="*",
            c="red",
            edgecolor="black",
            s=180,
            label=f"{self.n_sources} sources",
            zorder=6,
        )
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        return ax

from typing import Optional, Tuple

import math
import numpy as np

import matplotlib.pyplot as plt

from odil_wave.grid import Grid
from odil_wave.models.base import VelocityModel


class AcquisitionGeometry:
    """Elliptical array of transducers around the interior region.

    `n_receivers` transducer positions record every shot. A subset
    (`n_sources`, evenly spaced) act as shot sources.
    """

    def __init__(
        self,
        grid: Grid,
        n_receivers: int = 16,
        n_sources: Optional[int] = None,
        f0: float = 4.0,
        t0: Optional[float] = None,
        sigma_s: Optional[float] = None,
        a_frac: float = 0.55,
        b_frac: float = 0.70,
        ring_center: Tuple[float, float] = (0.0, 0.0),
    ):
        # inherit geometry from grid
        self.grid = grid

        self.n_receivers = n_receivers
        self.n_sources = n_receivers if n_sources is None else n_sources

        # source config
        self.f0 = f0  # peak frequency
        self.t0 = 1.0 / f0 if t0 is None else t0  # causal Ricker delay [s]
        self.sigma_s = (  # spatial spread of the source
            1.5 * max(grid.dx, grid.dy) if sigma_s is None else sigma_s
        )

        # ellipse config
        self.a_frac = a_frac  # aspect ratio
        self.b_frac = b_frac  # aspect ratio
        self.ring_center = ring_center

        # create geometry
        self.recv_ij = self._place_ellipse(self.n_receivers)
        step = max(1, self.n_receivers // self.n_sources)
        self.src_ij = self.recv_ij[::step][: self.n_sources]

    def _place_ellipse(self, n: int) -> np.ndarray:
        """Return (n, 2) integer full-grid indices on an ellipse inside the interior."""
        (ix_min, ix_max), (iy_min, iy_max) = self.grid.interior_extent
        cx, cy = self.ring_center
        a = self.a_frac * (ix_max - ix_min) / 2.0
        b = self.b_frac * (iy_max - iy_min) / 2.0

        k = np.arange(n)
        theta = 2.0 * math.pi * k / n
        x_k = cx + a * np.cos(theta)
        y_k = cy + b * np.sin(theta)

        (xmin, _), (ymin, _) = self.grid.extent
        i = np.round((x_k - xmin) / self.grid.dx).clip(0, self.grid.nx - 1).astype(int)
        j = np.round((y_k - ymin) / self.grid.dy).clip(0, self.grid.ny - 1).astype(int)

        return np.stack([i, j], axis=-1)

    def ricker(self, t: np.ndarray) -> np.ndarray:
        arg = (math.pi * self.f0 * (t - self.t0)) ** 2
        return (1.0 - 2.0 * arg) * np.exp(-arg)

    def src_position(self, src_idx: int) -> Tuple[float, float]:
        i, j = int(self.src_ij[src_idx, 0]), int(self.src_ij[src_idx, 1])
        return float(self.grid.x[i]), float(self.grid.y[j])

    def recv_position(self, rcv_idx: int) -> Tuple[float, float]:
        i, j = int(self.recv_ij[rcv_idx, 0]), int(self.recv_ij[rcv_idx, 1])
        return float(self.grid.x[i]), float(self.grid.y[j])

    def source_field(self, src_idx: int) -> np.ndarray:
        """(NT, NX, NY) Gaussian-in-space, Ricker-in-time source field."""
        x_src, y_src = self.src_position(src_idx)
        spatial = np.exp(
            -(
                ((self.grid.X - x_src) ** 2 + (self.grid.Y - y_src) ** 2)
                / self.sigma_s**2
            )
        )
        temporal = self.ricker(self.grid.t)

        # broadcast (Nt,) * (Nx, Ny) -> (Nt, 1, 1) * (1, Nx, Ny) => (Nt, Nx, Ny)
        return temporal.reshape(-1, 1, 1) * spatial.reshape(1, *self.grid.shape)

    def source_matrix(self) -> np.ndarray:
        """precompute source fields for each shot"""
        n_shots = self.n_sources

        sources = (
            np.stack([self.source_field(i) for i in range(n_shots)])
            .reshape(n_shots, -1)
            .T
        )  # (nt*nx*ny, n_shots)
        return sources

    def extract_observations(self, U: np.ndarray) -> np.ndarray:
        """Pull (NT, n_receivers) sensor data from a (NT, NX, NY) wavefield."""
        return U[:, self.recv_ij[:, 0], self.recv_ij[:, 1]]

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
        """Plot the acquisition geometry"""
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 5))
        velocity_model.show(
            ax=ax, title=f"Acquisition on {velocity_model.name}", show_pml=True
        )

        rx = self.grid.x[self.recv_ij[:, 0]]
        ry = self.grid.y[self.recv_ij[:, 1]]
        sx = self.grid.x[self.src_ij[:, 0]]
        sy = self.grid.y[self.src_ij[:, 1]]
        ax.scatter(
            rx,
            ry,
            marker="v",
            c="lime",
            edgecolor="black",
            s=70,
            label=f"{self.n_receivers} receivers",
            zorder=5,
        )
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

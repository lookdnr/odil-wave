import numpy as np
import matplotlib.pyplot as plt

from odil_wave.models.base import VelocityModel
from .sources import Sources
from .receivers import Receivers


class AcquisitionGeometry:
    """
    Convenience class for collecting Sources and Receivers into a unified object.

    TODO: Downstream functionality operates on AcquistionGeometry instances, need to
    check functionality interfacing
    """

    def __init__(self, sources: Sources, receivers: Receivers):
        self.sources = sources
        self.receivers = receivers

    def source_matrix(self) -> np.ndarray:
        return self.sources.source_matrix()

    def extract_observations(self, U: np.ndarray) -> np.ndarray:
        return self.receivers.extract_observations(U)

    def show(self, velocity_model: VelocityModel, ax=None, cmap: str | None = None):
        """Plot the acquisition geometry over the velocity model"""
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        if cmap is None:
            cmap = "RdBu_r"

        velocity_model.show(
            ax=ax,
            title=f"Acquisition | {velocity_model.name}",
            vmin=velocity_model.c_min,
            vmax=velocity_model.c_max,
            cmap=cmap,
        )

        x = self.sources.grid.x
        y = self.sources.grid.y

        recv_ij = self.receivers.recv_ij
        src_ij = self.sources.src_ij

        rx = x[recv_ij[:, 0]]
        ry = y[recv_ij[:, 1]]
        sx = x[src_ij[:, 0]]
        sy = y[src_ij[:, 1]]

        ax.scatter(
            sx,
            sy,
            marker="*",
            c="red",
            edgecolor="black",
            s=180,
            label="Sources",
        )

        ax.scatter(
            rx,
            ry,
            marker="v",
            c="lime",
            edgecolor="black",
            s=70,
            label="Receivers",
        )

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,
            frameon=False,
            fontsize=15,
        )
        ax.margins(0)
        return ax

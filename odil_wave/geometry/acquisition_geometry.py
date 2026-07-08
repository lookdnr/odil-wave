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

    def show(self, velocity_model: VelocityModel, ax=None):
        """Plot the acquisition geometry"""
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 5))
        velocity_model.show(ax=ax, title=f"Acquisition on {velocity_model.name}")
        x = self.sources.grid.x
        y = self.sources.grid.y

        recv_ij = self.receivers.recv_ij
        src_ij = self.sources.src_ij

        n_recv = self.receivers.n_receivers
        n_src = self.sources.n_sources

        rx = x[recv_ij[:, 0]]
        ry = y[recv_ij[:, 1]]
        sx = x[src_ij[:, 0]]
        sy = y[src_ij[:, 1]]
        ax.scatter(
            rx,
            ry,
            marker="v",
            c="lime",
            edgecolor="black",
            s=70,
            label=f"{n_recv} receivers",
            zorder=5,
        )
        ax.scatter(
            sx,
            sy,
            marker="*",
            c="red",
            edgecolor="black",
            s=180,
            label=f"{n_src} sources",
            zorder=6,
        )
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        return ax

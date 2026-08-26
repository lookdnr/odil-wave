import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes

from odil_wave.models.base import VelocityModel
from .sources import Sources
from .receivers import Receivers


class AcquisitionGeometry:
    """Convenience class for representing `Sources` and `Receivers` in a single object.

    Attributes
    ----------
    sources : Sources
        Source positions and wavelet injection.
    receivers : np.ndarray, optional
        Receiever positions and observation extraction.
    """

    def __init__(self, sources: Sources, receivers: Receivers):
        """Bundle a source and receiver layout into one acquisition geometry.

        Parameters
        ----------
        sources : Sources
            Source positions and wavelet injection.
        receivers : Receivers
            Receiver positions and trace extraction.
        """
        self.sources = sources
        self.receivers = receivers

    def source_matrix(self) -> np.ndarray:
        """Return the source injection matrix.

        Returns
        -------
        np.ndarray
            Source injection matrix.

        Notes
        -----
        Delegates to `Sources.source_matrix`.
        """
        return self.sources.source_matrix()

    def extract_observations(self, U: np.ndarray) -> np.ndarray:
        """Sample the wavefield at receiver locations

        Parameters
        ----------
        U : np.ndarray
            Full wavefield array.

        Notes
        -----
        Delegates to `Receivers.extract_observations`.
        """
        return self.receivers.extract_observations(U)

    def show(
        self,
        velocity_model: VelocityModel,
        ax: matplotlib.axes.Axes | None = None,
        cmap: str | None = None,
        cbar: bool = True,
    ):
        """Plot the acquisition geometry over the velocity model.

        Parameters
        ----------
        velocity_model : VelocityModel
            Velocity model used as the plots colour field.
        ax : matplotlib.ax.Axes or None
            Axes to draw on. A new figure is created if None.
        cmap : str or None
            Colour map for the velocity field (default "viridis").

        Returns
        -------
        matplotlib.axes.Axes
            The axes the geometry was plotted on.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        if cmap is None:
            cmap = "viridis"

        velocity_model.show(
            ax=ax,
            title=f"Acquisition | {velocity_model.name}",
            vmin=velocity_model.c_min,
            vmax=velocity_model.c_max,
            cmap=cmap,
            cbar=cbar,
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

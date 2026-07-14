from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.ticker import EngFormatter
from matplotlib.colors import SymLogNorm
from typing import Any

from odil_wave.grid import Grid
from odil_wave.models.base import VelocityModel


@dataclass
class Wavefield:
    grid: Grid
    _amplitude: np.ndarray = field(init=False)
    init_amplitude: np.ndarray | None = None  # optionally initialise amplitude

    def __post_init__(self) -> None:
        Nx, Ny = self.grid.shape
        Nt = self.grid.nt

        # initialise amplitude as Nx*Ny*Nt or provided values
        self._amplitude = (
            np.zeros(shape=(Nt, Nx * Ny))  # store as (time, space)
            if self.init_amplitude is None
            else np.asarray(self.init_amplitude).reshape(Nt, Nx * Ny)
        )

    @property
    def U(self) -> np.ndarray:
        return self._amplitude

    @U.setter
    def U(self, value: np.ndarray) -> None:
        Nt = self.grid.nt
        Nx, Ny = self.grid.shape
        self._amplitude = value.reshape(Nt, Nx * Ny)

    @property
    def flat_data(self) -> np.ndarray:
        """Return flat parameter vector amp (nt*nx*ny) as np.ndarray"""
        return self._amplitude.ravel()

    @flat_data.setter
    def flat_data(self, flat: np.ndarray) -> None:
        """Cast flattened data back into amplitude and wavespeed tensors"""
        Nx, Ny = self.grid.shape
        Nt = self.grid.nt

        # reshape
        try:
            self._amplitude = flat.reshape(Nt, Nx * Ny)
        except ValueError as e:
            raise ValueError(
                f"expected flat vector of length {Nt * Nx * Ny}, got {flat.size}"
            ) from e

    def show(self, idx: int, title="Wavefield", view: str = "xy"):
        assert view in [
            "xy",
            "ty",
            "tx",
        ], "View must be str and one of 'xy', 'ty', 'tx'"
        axis = {"xy": 0, "ty": 1, "tx": 2}[view]  # extract axis index

        # extrcat shapes
        Nx, Ny = self.grid.shape
        Nt = self.grid.nt

        param = [Nt, Nx, Ny][axis]  # extract upper limit of axis

        if not (0 <= idx < param):
            raise ValueError(f"idx should be an int in range [0, {param}], got {idx}.")

        amp_data = np.take(self._amplitude, idx, axis=axis).reshape(
            Nx, Ny
        )  # extract slice

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        (xmin, xmax), (ymin, ymax) = self.grid.extent

        im1 = ax.imshow(
            amp_data,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap="RdBu_r",
        )

        i, j = view[0], view[1]  # extract letters for labelling
        ax.set_xlabel(i)
        ax.set_ylabel(j)

        slice_plane = ["t", "x", "y"][axis]
        ax.set_title(f"Amplitude field ({view} plane, {slice_plane} = {idx})")

        plt.colorbar(im1, ax=ax, label="Amplitude", shrink=0.85)

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    def animate(
        self,
        filename: str = "wavefield.gif",
        fps: int = 20,
        cmap: str = "seismic",
        title: str = "Wavefield history",
        scaling: str | None = None,
        db_floor=-60.0,  # dynamic range for scaling="dB"
        model: VelocityModel | np.ndarray | None = None,
        model_levels: int | float = 1,
        model_colour: str = "k",
        model_alpha: float = 0.35,
        model_linewidth: float = 0.9,
        model_linestyle: str = "--",
    ) -> str:
        """Render the amplitude field over all time steps to an animated GIF.

        Optionally overlays dashed contours from a velocity model to give
        structural context without competing with the wavefield colours.

        model can be either:
        - a VelocityModel instance (uses model.c), or
        - a raw ndarray with shape (Nx, Ny) or (Ny, Nx).
        """
        if scaling not in [None, "dB", "SymLog"]:
            raise ValueError(
                f"arg `scaling` must be one of 'db', 'SymLog', got {scaling}"
            )

        amp = self._amplitude  # (Nt, Nx, Ny)
        Nt = self.grid.nt
        Nx, Ny = self.grid.shape
        (xmin, xmax), (ymin, ymax) = self.grid.extent
        t = self.grid.t

        u_max = float(np.abs(amp).max()) or 1.0
        data = amp

        # scaling == None...
        plot_kwargs: dict[str, Any] = dict(cmap=cmap, vmin=-u_max, vmax=u_max)
        cbar_label = "Amplitude"

        if scaling == "dB":
            data = 20.0 * np.log10(np.maximum(np.abs(amp), 1e-30) / u_max)
            data = np.clip(data, db_floor, 0.0)  # 60db range
            plot_kwargs = dict(cmap="magma", vmin=db_floor, vmax=0.0)
            cbar_label = "dB rel. max"
        else:
            norm = SymLogNorm(linthresh=1e-3 * u_max, vmin=-u_max, vmax=u_max)
            plot_kwargs = dict(cmap=cmap, norm=norm)
            cbar_label = "Amplitude"

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            data[0].reshape(Nx, Ny).T,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            animated=True,
            **plot_kwargs,
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fmt_t = EngFormatter(unit="s", places=2)
        ttl = ax.set_title(f"{title}  (t = {fmt_t(t[0])})")
        plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.85)

        if model is not None:
            if isinstance(model, np.ndarray):
                model_data = model
            else:
                model_data = getattr(model, "c", None)
                if model_data is None:
                    raise TypeError("model must be a VelocityModel or ndarray")
            model_arr = np.asarray(model_data)
            if model_arr.shape == (Nx, Ny):
                model_plot = model_arr.T
            elif model_arr.shape == (Ny, Nx):
                model_plot = model_arr
            else:
                raise ValueError(
                    "model must be a VelocityModel or ndarray with shape "
                    + f"{(Nx, Ny)} or {(Ny, Nx)}, got {model_arr.shape}"
                )

            mmin, mmax = float(model_plot.min()), float(model_plot.max())
            if mmax > mmin:
                if isinstance(model_levels, int):
                    if model_levels <= 1:
                        levels = [(mmin + mmax) * 0.5]
                    else:
                        levels = np.linspace(mmin, mmax, model_levels + 2)[1:-1]
                else:
                    levels = np.asarray(model_levels, dtype=float)

                x = np.linspace(xmin, xmax, Nx)
                y = np.linspace(ymin, ymax, Ny)
                ax.contour(
                    x,
                    y,
                    model_plot,
                    levels=levels,
                    colors=model_colour,
                    linewidths=model_linewidth,
                    linestyles=model_linestyle,
                    alpha=model_alpha,
                    zorder=3,
                )

        # update for drawing frames
        def update(frame: int):
            im.set_data(amp[frame].reshape(Nx, Ny).T)
            ttl.set_text(f"{title} (t = {fmt_t(t[frame])})")
            return im, ttl

        anim = animation.FuncAnimation(
            fig, update, frames=Nt, interval=1000 / fps, blit=False
        )
        anim.save(filename, writer=animation.PillowWriter(fps=fps))
        plt.close(fig)
        return filename


if __name__ == "__main__":
    grid = Grid()
    amp = np.random.rand(grid.nt, *grid.shape)
    u = Wavefield(grid, init_amplitude=amp)
    u.show(title="Test plot", idx=100)
    plt.show()

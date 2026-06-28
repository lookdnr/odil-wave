from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from odil_wave.grid import Grid


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
        cmap: str = "RdBu_r",
        title: str = "Wavefield history",
    ) -> str:
        """Render the amplitude field over all time steps to an animated GIF."""

        amp = self._amplitude  # (Nt, Nx, Ny)
        Nt = self.grid.nt
        Nx, Ny = self.grid.shape
        (xmin, xmax), (ymin, ymax) = self.grid.extent
        t = self.grid.t

        # fixed colour scale
        vmax = float(np.abs(amp).max()) or 1.0
        vmin = -vmax

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            amp[0].reshape(Nx, Ny).T,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            animated=True,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ttl = ax.set_title(f"{title}  (t = {t[0]:.3f} s)")
        plt.colorbar(im, ax=ax, label="Amplitude", shrink=0.85)

        # update for drawing frames
        def update(frame: int):
            im.set_data(amp[frame].reshape(Nx, Ny).T)
            ttl.set_text(f"{title}  (t = {t[frame]:.3f} s)")
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

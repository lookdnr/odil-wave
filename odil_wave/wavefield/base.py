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

    _wavespeed: np.ndarray = field(init=False)
    init_wavespeed: np.ndarray | None = None  # optional initialise speed

    _init_ut: np.ndarray = field(init=False)
    init_velocity: np.ndarray | None = None  # optional initial velocity field

    def __post_init__(self) -> None:
        Nx, Ny = self.grid.shape
        Nt = self.grid.nt

        # initialise amplitude as Nx*Ny*Nt or provided values
        self._amplitude = (
            np.zeros(shape=(Nt, Nx, Ny))
            if self.init_amplitude is None
            else np.asarray(self.init_amplitude)
        )

        # initialise wavespeed as Nx*Ny or provided values
        self._wavespeed = (
            np.ones(shape=(Nx, Ny))
            if self.init_wavespeed is None
            else np.asarray(self.init_wavespeed)
        )

        # initialise velocity
        self._init_ut = (
            np.zeros(shape=(Nx, Ny))
            if self.init_velocity is None  # default init is zeros
            else np.asarray(self.init_velocity)
        )

    @property
    def init_ut(self) -> np.ndarray:
        return self._init_ut

    @property
    def amplitude(self) -> np.ndarray:
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: np.ndarray) -> None:
        Nt = self.grid.nt
        Nx, Ny = self.grid.shape
        self._amplitude = value.reshape(Nt, Nx, Ny)

    @property
    def wavespeed(self) -> np.ndarray:
        return self._wavespeed

    @property
    def flat_data(self) -> np.ndarray:
        """Return flat parameter vector [amp (nt*nx*ny), wvsp (nx*ny)] as np.ndarray"""
        return np.concat([self._amplitude.ravel(), self._wavespeed.ravel()])

    @flat_data.setter
    def flat_data(self, flat: np.ndarray) -> None:
        """Cast flattened data back into amplitude and wavespeed tensors"""
        Nx, Ny = self.grid.shape
        Nt = self.grid.nt

        n_amp = Nx * Ny * Nt  # num amplitude entries
        n_wsp = Nx * Ny  # num wavespeed entries
        total = n_amp + n_wsp  # total

        if flat.shape != (total,):
            raise ValueError(
                f"expected flat vector of length {total}, got {flat.shape}"
            )

        # resahpe and cast to np tensors
        self._amplitude = flat[:n_amp].reshape(Nt, Nx, Ny)
        self._wavespeed = flat[n_amp:].reshape(Nx, Ny)

    def show(self, idx: int, title="Wavefield and model", view: str = "xy"):
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

        amp_data = np.take(self.amplitude, idx, axis=axis)  # extract slice

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        (xmin, xmax), (ymin, ymax) = self.grid.extent

        im1 = axs[0].imshow(
            amp_data,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap="RdBu_r",
        )

        im2 = axs[1].imshow(
            self.wavespeed,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap="viridis",
        )

        i, j = view[0], view[1]  # extract letters for labelling
        for ax in axs:
            ax.set_xlabel(i)
            ax.set_ylabel(j)

        slice_plane = ["t", "x", "y"][axis]
        axs[0].set_title(f"Amplitude field ({view} plane, {slice_plane} = {idx})")
        axs[1].set_title("Wave speed model")

        plt.colorbar(im1, ax=axs[0], label="Amplitude", shrink=0.85)
        plt.colorbar(im2, ax=axs[1], label=r"Wavespeed ($ms^{-1}$)", shrink=0.85)

        fig.suptitle(title)
        fig.tight_layout()
        plt.show()

    def animate(
        self,
        filename: str = "wavefield.gif",
        fps: int = 20,
        cmap: str = "RdBu_r",
        title: str = "Wavefield history",
    ) -> str:
        """Render the amplitude field over all time steps to an animated GIF."""

        amp = self.amplitude  # (Nt, Nx, Ny)
        Nt = self.grid.nt
        (xmin, xmax), (ymin, ymax) = self.grid.extent
        t = self.grid.t

        # fixed colour scale
        vmax = float(np.abs(amp).max()) or 1.0
        vmin = -vmax

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            amp[0],
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
            im.set_data(amp[frame])
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
    wsp = np.random.rand(*grid.shape)
    u = Wavefield(grid, init_amplitude=amp, init_wavespeed=wsp)
    u.show(title="Test plot", idx=100)

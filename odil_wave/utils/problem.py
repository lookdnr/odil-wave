from dataclasses import dataclass

from odil_wave import WaveEquation, AcquisitionGeometry, Wavefield


@dataclass
class Problem:
    """Configuration for the loss function."""

    wave_eq: WaveEquation
    geometry: AcquisitionGeometry

    def __post_init__(self):
        wf = self.wave_eq.wavefield
        (self.Nx, self.Ny), self.Nt = wf.grid.shape, wf.grid.nt
        self.sources = (
            self.geometry.source_matrix()
        )  # precompute (nt*nx*ny, n_shots) source matrix

    @property
    def wavefield(self) -> Wavefield:
        return self.wave_eq.wavefield

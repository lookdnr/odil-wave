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

        # Time rows 0 and 1 are pinned by the initial conditions
        # (u(0)=0, u_t(0)=0), so the RHS in those rows must be zero. The source
        # field otherwise corrupts the ICs.
        ns = self.Nx * self.Ny
        self.sources[: 2 * ns, :] = 0.0

    @property
    def wavefield(self) -> Wavefield:
        return self.wave_eq.wavefield

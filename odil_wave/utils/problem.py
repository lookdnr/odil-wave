from dataclasses import dataclass

from odil_wave import WaveEquation, AcquisitionGeometry, Wavefield
from odil_wave.geometry import Sources


@dataclass
class Problem:
    """Configuration for the loss function.

    Bundles the discrete wave equation with an acquisition geometry (or
    a bare `Sources` object) and precomputes the flattened source matrix
    used by `DiscreteLoss` subclasses, with initial condition rows zeroed.

    Parameters
    ----------
    wave_eq : WaveEquation
        Discrete wave equation operator.
    geometry : AcquisitionGeometry or Sources
        Acquisition geometry (or just sources) supplying the source matrix.

    Attributes
    ----------
    Nx, Ny, Nt : int
        Spatial and temporal grid dimensions, taken from
        `wave_eq.wavefield.grid`.
    sources : np.ndarray
        (nt*nx*ny, n_shots) source matrix, with the first two time rows
        zeroed to respect the initial conditions u(0) = u_t(0) = 0.
    """

    wave_eq: WaveEquation
    geometry: AcquisitionGeometry | Sources

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
        """Wavefield: The wavefield attached to `wave_eq`."""
        return self.wave_eq.wavefield

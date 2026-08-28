import numpy as np
from typing import Tuple

from odil_wave import Grid
from .utils import build_weight_matrix, place_ellipse


class Receivers:
    """Create an array of receivers on a Grid.

    Two placement modes are suppoted: `"custom"` places receivers at
    explicit spatial coordinates, `"ring"` places them on an ellipse
    controlled by `ring_centre` and `a_frac`/ `b_frac`. Extraction
    from a wavefield uses Kaiser-windowed sinc interpolation over an
    `n_sinc` window in each spatial direction.

    Parameters
    ----------
    grid : Grid
        Grid the receivers are placed on.
    mode : {"custom", "ring"}
        Placement mode.
    receiver_locs : tuple of (float, float), optional
        Explicit (x, y) receiver coordinates, required if `mode` is "custom".
    n_receivers : int, optional
        Number of receivers to place on the ring, ignored if `mode` is "custom".
    a_frac, b_frac : float, optional
        Semi axis fractions of the ellipse for ring placement (mode="ring" only).
    ring_centre : tuple of (float, float), optional
        Centre of the receiver ring in spatial coordinates (mode="ring" only).
    n_sinc : int, optional
        Width of the sinc interpolation window used for extraction.

    Attributes
    ----------
    recv_xy : np.ndarray
        (n_receivers, 2) array of receiver coordinates.
    recv_ij : np.ndarray
        (n_receivers, 2) array of nearest-node grid indices, for display/indexing.
    W : np.ndarray
        (nx*ny, n_receivers) sinc interpolation weight matrix.

    Raises
    ------
    ValueError
        If `n_receivers` is not positive, `mode` is invalid, `receiver_locs`
        is missing under "custom" mode, or any location/centre falls outside
        the grid extent.
    """

    def __init__(
        self,
        grid: Grid,
        mode: str = "custom",
        receiver_locs: Tuple[Tuple[float, float], ...] | None = None,
        n_receivers: int = 16,
        a_frac: float = 0.55,
        b_frac: float = 0.70,
        ring_centre=(0.0, 0.0),
        n_sinc=8,
    ):
        # inherit geometry from grid
        self.grid = grid

        if n_receivers is None:
            n_receivers = 1

        # validate source input
        if not n_receivers > 0:
            raise ValueError(
                f"arg n_receivers must be greater than 0, got {n_receivers}"
            )

        self.n_receivers = n_receivers

        # extract extent for validity checking
        (x_min, x_max), (y_min, y_max) = grid.extent

        # validate mode
        self.mode = mode.strip().lower()

        if self.mode not in ("custom", "ring"):
            raise ValueError(f"arg mode must be one of 'custom' 'ring', got {mode}")

        if self.mode == "ring":
            # validate a_frac, b_frac input
            if a_frac is None or a_frac <= 0:
                raise ValueError(f"arg a_frac should be > 0, got {a_frac}")

            if b_frac is None or b_frac <= 0:
                raise ValueError(f"arg b_frac should be > 0, got {b_frac}")

            self.a_frac = a_frac  # aspect ratio
            self.b_frac = b_frac  # aspect ratio

            x, y = ring_centre

            # validate ring_centre input
            if not (x_min < x < x_max):
                raise ValueError(
                    f"x component of arg ring_centre must be in range {x_min, x_max}"
                    f" for the given Grid, got {x}."
                )

            if not (y_min < y < y_max):
                raise ValueError(
                    f"y component of arg ring_centre must be in range {y_min, y_max}"
                    f" for the given Grid, got {y}."
                )

            self.ring_centre = ring_centre

            self.recv_ij = place_ellipse(grid, n_receivers, ring_centre, a_frac, b_frac)

            # physical coords for sinc injection
            self.recv_xy = np.array(
                [[grid.x[i], grid.y[j]] for i, j in self.recv_ij], dtype=float
            )

        if self.mode == "custom":
            if receiver_locs is None:
                raise ValueError(
                    "arg receiver_locs cannot be None with mode == 'custom'."
                )

            # validate source locs
            for k, (x, y) in enumerate(receiver_locs):
                if not (x_min < x < x_max):
                    raise ValueError(
                        f"x component of receiver_locs must be in range {x_min, x_max}"
                        f" for the given Grid, got {x} for receiver location {k}."
                    )

                if not (y_min < y < y_max):
                    raise ValueError(
                        f"y component of receiver_locs must be in range {y_min, y_max}"
                        f" for the given Grid, got {y} for receiver location {k}."
                    )

            self.recv_xy = np.array(receiver_locs, dtype=float)  # (n_sources, 2)
            # snap to nearest node for display / indexing
            i = (
                np.round((self.recv_xy[:, 0] - x_min) / grid.dx)
                .clip(0, grid.nx - 1)
                .astype(int)
            )
            j = (
                np.round((self.recv_xy[:, 1] - y_min) / grid.dy)
                .clip(0, grid.ny - 1)
                .astype(int)
            )
            self.recv_ij = np.stack([i, j], axis=-1)

            self.n_receivers = self.recv_xy.shape[0]

        self.W = build_weight_matrix(
            grid, self.recv_xy, self.n_receivers, n_sinc
        )  # (nx*ny, n_recv)

    def extract_observations(self, U: np.ndarray) -> np.ndarray:
        """Extract wavefield observations at receiver locations.

        Parameters
        ----------
        U : np.ndarray
            Full wavefield array.

        Returns
        -------
        np.ndarray
            Array of traces extracted at receiver locations.

        Notes
        -----
        Uses Kaiser-windowed sinc interpolation configured at construction.
        """
        nt = U.shape[0]
        return U.reshape(nt, -1) @ self.W  # (nt, n_recv)

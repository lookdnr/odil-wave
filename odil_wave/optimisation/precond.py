import numpy as np
from scipy.sparse.linalg import splu, LinearOperator

from odil_wave.operator.wave import WaveEquation


class AlphaCirculantPreconditioner:
    """alpha-circulant (ParaDiag-II) preconditioner for the reduced system.

    Approximates the BTTB operator by wrapping its time stencil
    around the corner, damped by alpha.

    The wrap makes it block-circulant, and hence invertible by FFT-in-time
    + one small spatial solve per mode.

    Differs from the true operator only in two corner block rows.

    alpha trades approximation error against taper roundoff
    """

    def __init__(
        self, blocks, n, alpha=1e-3, dtype: np.typing.DTypeLike = np.complex128
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                f"alpha must be in (0, 1), got {alpha}. alpha >= 1 is the "
                + "undamped time-periodic operator, which is singular "
                + "for wave problems."
            )
        if any(np.iscomplexobj(B.data) for B in blocks):
            raise ValueError("blocks must be real (conjugate-pair solve assumes it)")

        self.blocks, self.n, self.alpha = list(blocks), n, alpha
        self.ns = blocks[0].shape[0]

        # set up the taper to undo the periodicity
        # this is basically an absorbing layer in time
        gamma = alpha ** (1.0 / n)
        self._d = gamma ** np.arange(n)
        z = gamma * np.exp(-2j * np.pi * np.arange(n // 2 + 1) / n)

        self.dtype = dtype

        # LU decomposition paid once for solve at each frequency
        # we factorise each block to solve the N independent scalar equations
        # for each time level

        # note we only factorise the first half of the modes, since the FFT of
        # real data is conjugate symmetric and the remaining LU factors are
        # simply the complex conjugates of these ones
        # this cuts our memory requiremetns in half
        B0, B1, B2 = self.blocks
        self._lus = [
            splu(
                (B0 + zk * B1 + zk**2 * B2).astype(self.dtype).tocsc(),
                permc_spec="MMD_AT_PLUS_A",
                options=dict(SymmetricMode=True, DiagPivotThresh=0.001),
            )
            for zk in z
        ]

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """Compute the action of the preconditioner on a vector v"""
        V = v.reshape(self.n, self.ns) * self._d[:, None]

        # perform fft in time
        # rfft is for real valued inputs
        Vh = np.fft.rfft(V, axis=0).astype(self.dtype)

        # in the Fourier basis, the linear solve reduces
        # to solving N scalar equations for a block in time
        # we solve each time block below
        Wh = np.empty_like(Vh)
        for k in range(Vh.shape[0]):
            Wh[k] = self._lus[k].solve(Vh[k])  # solve the LU factorisation for du

        W = np.fft.irfft(Wh, n=self.n, axis=0) / self._d[:, None]  # transform back
        return W.ravel()

    def as_linear_operator(self) -> LinearOperator:
        """Return the preconditioner as a scipy.sparse.linalg.LinearOperator"""
        N = self.n * self.ns
        return LinearOperator(
            shape=(N, N), matvec=self.matvec, dtype=np.float64  # type: ignore
        )

    @classmethod
    def from_wave_equation(cls, we: WaveEquation, alpha=1e-3):
        """Build the preconditioner from a WaveEquation object"""
        return cls(we.reduced_blocks, we.nt - 2, alpha)

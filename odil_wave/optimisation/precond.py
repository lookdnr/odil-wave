import numpy as np
from scipy.sparse.linalg import splu, LinearOperator

from odil_wave.operator.wave import WaveEquation


class AlphaCirculantPreconditioner:
    """alpha-circulant (ParaDiag-II) preconditioner for the reduced system.

    Approximates the BTTB operator by wrapping its time stencil
    around the corner, damped by alpha.

    The wrap makes it block-circulant, and hence invertible by FFT-in-time
    + one small spatial solve per mode.

    Differs from the true operator only in two corner block-rows.

    alpha trades approximation error against taper roundoff
    """

    def __init__(self, blocks, n, alpha=1e-3):
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

        gamma = alpha ** (1.0 / n)
        self._d = gamma ** np.arange(n)
        z = gamma * np.exp(-2j * np.pi * np.arange(n) / n)

        # LU decomposition paid once for solve at each frequency
        self._lus = [
            splu(sum(zk**idx * B for idx, B in enumerate(self.blocks)))
            .astype(np.complex128)
            .tocsr()
            for zk in z[: n // 2 + 1]
        ]

    def matvec(self, v: np.ndarray) -> np.ndarray:
        V = v.reshape(self.n, self.ns) * self._d[:, None]
        Vh = np.fft.fft(V, axis=0)  # fft in time

        Wh = np.empty_like(Vh)
        for k in range(self.n // 2 + 1):
            Wh[k] = self._lus[k].solve(Vh[k])

        for k in range(self.n // 2 + 1, self.n):
            Wh[k] = np.conj(self._lus[self.n - k].solve(np.conj(Vh[k])))

        W = np.fft.ifft(Wh, axis=0) / self._d[:, None]  # transform back
        return W.real.ravel()

    def as_linear_operator(self) -> LinearOperator:
        N = self.n * self.ns
        return LinearOperator(
            shape=(N, N), matvec=self.matvec, dtype=np.float64  # type: ignore
        )

    @classmethod
    def from_wave_equation(cls, we: WaveEquation, alpha=1e-3):
        return cls(we.reduced_blocks, we.nt - 2, alpha)

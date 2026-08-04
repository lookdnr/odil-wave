import numpy as np
import threadpoolctl
import scipy.sparse.linalg as sl
from typing import List

import multiprocessing as mpl
from multiprocessing.connection import Connection

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
        self,
        blocks,
        n,
        alpha=1e-3,
        dtype: np.typing.DTypeLike = np.complex64,
        cache_factors: bool = True,
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
        # note we only need n/2n+1 values since the data is real-valued,
        # meaning we have complex conjugacy and need only ever other mode
        # np.rfft bakes this logic in
        gamma = alpha ** (1.0 / n)
        self._d = gamma ** np.arange(n)
        self._z = gamma * np.exp(-2j * np.pi * np.arange(n // 2 + 1) / n)

        self.dtype = dtype

        self._cache_factors = cache_factors
        self._lus = None  # list of factorisations

    def _factorise_mode(self, zk: float) -> sl.SuperLU:
        """Factorise a circulant block for a single mode z_k"""
        B0, B1, B2 = self.blocks
        Az = (B0 + zk * B1 + zk**2 * B2).astype(self.dtype).tocsc()
        return sl.splu(  # factorise
            Az,
            permc_spec="MMD_AT_PLUS_A",  # optimal for our structure
            options=dict(SymmetricMode=True, DiagPivotThresh=0.001),
        )

    def _factors(self) -> List[sl.SuperLU]:
        """Per mode LU factorisations, cached"""
        if self._lus is None:
            with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
                self._lus = [self._factorise_mode(zk) for zk in self._z]
        return self._lus

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

        # splu runs BLAS under the hood
        # measurements indicate pinning BLAS to a single thread leads to
        # far better perfromance for the factorisation stage
        blas_off = threadpoolctl.threadpool_limits(limits=1, user_api="blas")

        with blas_off:
            if self._cache_factors:  # cached branch
                lus = self._factors()
                for k in range(len(self._z)):
                    Wh[k] = lus[k].solve(Vh[k])  # solve the kth mode

            else:  # on-the-fly
                for k, zk in enumerate(self._z):
                    lu = self._factorise_mode(zk)
                    Wh[k] = lu.solve(Vh[k])  # solve for du
                    del lu  # discard to avoid memory blow up

        W = np.fft.irfft(Wh, n=self.n, axis=0) / self._d[:, None]  # transform back
        return W.ravel()

    def as_linear_operator(self) -> sl.LinearOperator:
        """Return the preconditioner as a scipy.sparse.linalg.LinearOperator"""
        N = self.n * self.ns
        return sl.LinearOperator(
            shape=(N, N), matvec=self.matvec, dtype=np.float64  # type: ignore
        )

    @classmethod
    def from_wave_equation(
        cls, we: WaveEquation, alpha=1e-3, cache_factors: bool = True
    ):
        """Build the preconditioner from a WaveEquation object"""
        return cls(we.reduced_blocks, we.nt - 2, alpha, np.complex128, cache_factors)


def _worker(worker_endpoint: Connection, blocks, z_local, dtype):
    """Task delegated to a worker in the forked process pool:
    factorise and solve allocated modes for the given circulant blocks
    """
    blas_off = threadpoolctl.threadpool_limits(limits=1, user_api="blas")
    B0, B1, B2 = blocks

    with blas_off:
        lus = [
            sl.splu(
                (B0 + zk * B1 + zk**2 * B2).astype(dtype).tocsc(),
                permc_spec="MMD_AT_PLUS_A",
                options=dict(SymmetricMode=True, DiagPivotThresh=0.001),
            )
            for zk in z_local  # factorise my modes
        ]

    # alert ready to receive
    worker_endpoint.send("ready")

    # recv fourier components to solve with
    while True:  # worker survives indefinitely

        vh_local = worker_endpoint.recv()  # blocks until received

        # exit flag
        if vh_local is None:
            return

        # send solutions
        worker_endpoint.send(
            np.stack([lu.solve(vh_local[i]) for i, lu in enumerate(lus)])
        )


class ParallelAlphaCirculantPreconditioner:
    """alpha-circulant (ParaDiag-II) preconditioner for the reduced system.

    Approximates the BTTB operator by wrapping its time stencil
    around the corner, damped by alpha.

    The wrap makes it block-circulant, and hence invertible by FFT-in-time
    + one small spatial solve per mode.

    Differs from the true operator only in two corner block rows.

    Employs `multiprocessing` Process workers connected by Pipes to distribute
    the circulant block factorisations, since they are entirely separable.
    Each worker factorises its partition, solves its factorisations, and
    sends its solution blocks.
    """

    def __init__(
        self,
        blocks,
        n,
        n_workers: int,
        alpha=1e-3,
        dtype: np.typing.DTypeLike = np.complex64,
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
        # note we only need n/2n+1 values since the data is real-valued,
        # meaning we have complex conjugacy and need only ever other mode
        # np.rfft bakes this logic in
        gamma = alpha ** (1.0 / n)
        self._d = gamma ** np.arange(n)
        self._z = gamma * np.exp(-2j * np.pi * np.arange(n // 2 + 1) / n)

        self.dtype = dtype

        # split up work by partitioning modes into subarrays
        parts = np.array_split(np.arange(len(self._z)), n_workers)

        # partitition work
        self._parts, self._home_endpoints, self._procs = [], [], []
        for part in parts:
            home_end, worker_end = mpl.Pipe()  # endpoints of communication

            # spawn worker child from this entrypoint
            # each child gets a copy of the provided parent's memory and hits the
            # target functions
            proc = mpl.Process(
                target=_worker, args=(worker_end, blocks, self._z[part], dtype)
            )

            proc.start()  # fork all procs now

            self._parts.append(part)
            self._home_endpoints.append(home_end)
            self._procs.append(proc)

        # receive results
        # multiproc equivalent of mpi barrier
        for home_end in self._home_endpoints:
            home_end.recv()

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """Compute the action of the preconditioner on a vector v"""
        V = v.reshape(self.n, self.ns) * self._d[:, None]

        # perform fft in time
        # rfft is for real valued inputs
        Vh = np.fft.rfft(V, axis=0).astype(self.dtype)

        # distribute work from endpoints
        # "scatter"
        for endpoint, part in zip(self._home_endpoints, self._parts):
            endpoint.send(Vh[part])  # scatter parts to workers

        # solution field
        Wh = np.empty_like(Vh)

        # receive solution parts to endpoints
        # "gather"
        for endpoint, part in zip(self._home_endpoints, self._parts):
            Wh[part] = endpoint.recv()  # gather solution parts from workers

        W = np.fft.irfft(Wh, n=self.n, axis=0) / self._d[:, None]  # transform back
        return W.ravel()

    def as_linear_operator(self) -> sl.LinearOperator:
        """Return the preconditioner as a scipy.sparse.linalg.LinearOperator"""
        N = self.n * self.ns
        return sl.LinearOperator(
            shape=(N, N), matvec=self.matvec, dtype=np.float64  # type: ignore
        )

    @classmethod
    def from_wave_equation(cls, we: WaveEquation, n_workers, alpha=1e-3):
        """Build the preconditioner from a WaveEquation object"""
        return cls(we.reduced_blocks, we.nt - 2, n_workers, alpha, np.complex128)

    def shutdown(self):
        """Shutdown the workers in the parallel pool"""
        for endpoint in self._home_endpoints:
            endpoint.send(None)

        for p in self._procs:
            p.join()

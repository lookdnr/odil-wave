import numpy as np
import threadpoolctl
import scipy.sparse.linalg as sl
from typing import List
import os
import warnings

import multiprocessing as mpl
from multiprocessing.connection import Connection

from odil_wave.operator.wave import WaveEquation


class AlphaCirculantPreconditioner:
    """alpha-circulant (ParaDiag-II) preconditioner for the reduced BTTB system.

    Approximates the Block-Toeplitz-with-Toeplitz-Blocks (BTTB) operator
    by wrapping its time stencil around the corner, damped by `alpha`.
    The wrap makes the operator block-circulant and hence invertible by
    FFT-in-time plus one spatial factorisation/solve per temporal
    Fourier mode. Differs from the true operator only in two corner
    block rows.

    Parameters
    ----------
    blocks : tuple of scipy.sparse.csr_array
        Time stencil blocks (B0, B1, B2) of the reduced system, as
        returned by `WaveEquation.reduced_blocks`.
    n : int
        Number of time steps in the reduced system.
    alpha : float, optional
        Damping parameter in (0, 1). alpha >= 1 gives the undamped,
        operator.
    dtype : numpy dtype, optional
        Complex dtype used for the per-mode factorisations.
    cache_factors : bool, optional
        If True, cache per-mode LU factorisations after first use. If
        False, factorise and discard on every `matvec` call.

    Raises
    ------
    ValueError
        If `alpha` is not in (0, 1), or `blocks` contains complex data.
    """

    def __init__(
        self,
        blocks,
        n,
        alpha=1e-3,
        dtype: np.typing.DTypeLike = np.complex128,
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
        """Apply the preconditioner to a vector.

        Parameters
        ----------
        v : np.ndarray
            Flattened (n*ns,) vector to precondition.

        Returns
        -------
        np.ndarray
            Flattened (n*ns,) preconditioned vector.
        """
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
        """Wrap this preconditioner as a `scipy.sparse.linalg.LinearOperator`.

        Returns
        -------
        scipy.sparse.linalg.LinearOperator
            Operator suitable for use as the `M` argument to
            `scipy.sparse.linalg.gmres`.
        """
        N = self.n * self.ns
        return sl.LinearOperator(
            shape=(N, N), matvec=self.matvec, dtype=np.float64  # type: ignore
        )

    @classmethod
    def from_wave_equation(
        cls,
        we: WaveEquation,
        alpha=1e-3,
        dtype: np.typing.DTypeLike = np.complex128,
        cache_factors: bool = True,
    ):
        """Build the preconditioner directly from a `WaveEquation`.

        Parameters
        ----------
        we : WaveEquation
            Wave equation to build the preconditioner for.
        alpha : float, optional
            Damping parameter in (0, 1).
        dtype : numpy dtype, optional
            Complex dtype used for the per-mode factorisations.
        cache_factors : bool, optional
            If True, cache per-mode LU factorisations after first use.

        Returns
        -------
        AlphaCirculantPreconditioner
            Preconditioner for `we`.
        """
        return cls(we.reduced_blocks, we.nt - 2, alpha, dtype, cache_factors)


def _caching_worker(worker_endpoint: Connection, blocks, z_local, dtype):
    """Task delegated to a worker in the forked process pool:
    factorise and solve allocated modes for the given circulant blocks.

    The caching worker computes all of its factorisations up front
    before waiting to receieves their RHS to solve with. This is the most
    efficient usage, but costs dearly in memory.
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


def _non_caching_worker(worker_endpoint: Connection, blocks, z_local, dtype):
    """Task delegated to a worker in the forked process pool:
    factorise and solve allocated modes for the given circulant blocks.

    The non-caching worker computes their factorisations 1 by 1, solving
    and discarding each factorisation after each solve. This is the
    memory-constrained case.
    """

    blas_off = threadpoolctl.threadpool_limits(limits=1, user_api="blas")
    B0, B1, B2 = blocks

    def factorise_one(zk):
        with blas_off:
            Az = (B0 + zk * B1 + zk**2 * B2).astype(dtype).tocsc()
            return sl.splu(
                Az,
                permc_spec="MMD_AT_PLUS_A",
                options=dict(SymmetricMode=True, DiagPivotThresh=0.001),
            )

    # alert ready to receive
    worker_endpoint.send("ready")

    # recv fourier components to solve with
    while True:  # worker survives indefinitely

        vh_local = worker_endpoint.recv()  # blocks until received

        # exit flag
        if vh_local is None:
            return

        result = np.empty_like(vh_local)
        for i, zk in enumerate(z_local):
            lu = factorise_one(zk)
            with blas_off:
                result[i] = lu.solve(vh_local[i])
            del lu

        # send solutions
        worker_endpoint.send(result)


class ParallelAlphaCirculantPreconditioner:
    """alpha-circulant (ParaDiag-II) preconditioner, parallelised across processes.

    Same mathematical construction as `AlphaCirculantPreconditioner`, but
    distributes the per-mode circulant block factorisations across
    `multiprocessing` worker processes connected by pipes, since the
    modes are entirely separable. Each worker factorises and solves its
    partition of modes and returns its solution blocks.

    Parameters
    ----------
    blocks : tuple of scipy.sparse.csr_array
        Time-stencil blocks (B0, B1, B2) of the reduced system, as
        returned by `WaveEquation.reduced_blocks`.
    n : int
        Number of time steps in the reduced system.
    n_workers : int
        Requested number of worker processes. Clamped to the number of
        available CPU cores and the number of Fourier modes, whichever
        is smaller.
    alpha : float, optional
        Damping parameter in (0, 1). alpha >= 1 gives the undamped operator.
    dtype : numpy dtype, optional
        Complex dtype used for the per-mode factorisations.
    cache_factors : bool, optional
        If True, each worker caches its factorisations up front (faster,
        more memory). If False, factorises and discards per mode.

    Raises
    ------
    ValueError
        If `alpha` is not in (0, 1), or `blocks` contains complex data.

    Warns
    -----
    UserWarning
        If `n_workers` exceeds the available cores or the number of
        modes, the pool size is clamped down in that case.
    """

    def __init__(
        self,
        blocks,
        n,
        n_workers: int,
        alpha=1e-3,
        dtype: np.typing.DTypeLike = np.complex128,
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

        self.caching = cache_factors

        n_cores = len(os.sched_getaffinity(0))
        n_modes = len(self._z)

        if n_workers > n_cores:
            warnings.warn(
                f"{n_workers} requested, only found {n_cores} valid cores."
                + f" Executing with {n_cores} processes in the pool.\n",
                UserWarning,
            )

        if n_workers > n_modes:
            warnings.warn(
                f"{n_workers} requested, only {n_modes} require factorising."
                + f" Executing with {n_modes} processes in the pool.\n",
                UserWarning,
            )

        # take the number of workers to be the minimum of
        # requested num_workers, number of cores, and number of modes
        # we are solving for
        # this avoids oversubscription
        n_workers = min(n_workers, n_cores, n_modes)

        # split up work by partitioning modes into subarrays
        parts = np.array_split(np.arange(len(self._z)), n_workers)

        worker = _caching_worker if self.caching else _non_caching_worker

        # partitition work
        self._parts, self._home_endpoints, self._procs = [], [], []
        for part in parts:
            home_end, worker_end = mpl.Pipe()  # endpoints of communication

            # spawn worker child from this entrypoint
            # each child gets a copy of the provided parent's memory and hits the
            # target functions
            proc = mpl.Process(
                target=worker, args=(worker_end, blocks, self._z[part], dtype)
            )
            proc.daemon = True  # forces children to shutdown if parent dies

            proc.start()  # fork all procs now

            self._parts.append(part)
            self._home_endpoints.append(home_end)
            self._procs.append(proc)

        # receive results
        # multiproc equivalent of mpi barrier
        for home_end in self._home_endpoints:
            home_end.recv()

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """Apply the preconditioner by scattering Fourier modes to workers.

        Parameters
        ----------
        v : np.ndarray
            Flattened (n*ns,) vector to precondition.

        Returns
        -------
        np.ndarray
            Flattened (n*ns,) preconditioned vector.
        """
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
        """Wrap this preconditioner as a `scipy.sparse.linalg.LinearOperator`.

        Returns
        -------
        scipy.sparse.linalg.LinearOperator
            Operator suitable for use as the `M` argument to
            `scipy.sparse.linalg.gmres`.
        """
        N = self.n * self.ns
        return sl.LinearOperator(
            shape=(N, N), matvec=self.matvec, dtype=np.float64  # type: ignore
        )

    @classmethod
    def from_wave_equation(
        cls,
        we: WaveEquation,
        n_workers: int,
        alpha=1e-3,
        dtype: np.typing.DTypeLike = np.complex128,
        caching: bool = True,
    ):
        """Build the preconditioner directly from a `WaveEquation`.

        Parameters
        ----------
        we : WaveEquation
            Wave equation to build the preconditioner for.
        n_workers : int
            Requested number of worker processes.
        alpha : float, optional
            Damping parameter in (0, 1).
        dtype : numpy dtype, optional
            Complex dtype used for the per-mode factorisations.
        caching : bool, optional
            If True, each worker caches its factorisations up front.

        Returns
        -------
        ParallelAlphaCirculantPreconditioner
            Preconditioner for `we`'s reduced system.
        """
        return cls(we.reduced_blocks, we.nt - 2, n_workers, alpha, dtype, caching)

    def shutdown(self):
        """Terminate the worker pool.

        Sends a stop signal to each worker and joins its process,
        force terminating any that don't exit within 5 seconds.
        """
        for endpoint in self._home_endpoints:
            try:
                endpoint.send(None)  # kill while True worker loop with None flag

            except (BrokenPipeError, OSError):  # e.g., worker already dead
                pass

        for p in self._procs:
            p.join(timeout=5)  # wait for a little

            if p.is_alive():
                p.terminate()  # kill any that don't shutdown gracefully

        self._closed = True

    def __enter__(self):
        """Entry context manager for 'with' blocks"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager for 'with' blocks"""
        self.shutdown()
        return False

from .base import Optimiser
from odil_wave.wavefield import Wavefield
from odil_wave.loss import DiscreteLoss
from .utils import create_u0, CountedOperator
from .precond import AlphaCirculantPreconditioner
from odil_wave.metrics import SolveRecorder, InnerRecord

from typing import Dict, Callable
import numpy as np
import scipy.sparse.linalg as spl
from time import perf_counter


class GaussNewtonOptimiser(Optimiser):
    """Gauss Newton Optimiser. Approximates the Hessian by H ~ J^T J, where
    J is the Jacobian of the functional, then solves:

        - J^T J du = -J^t r for du (matrix-free, choice of solve method)
        - u_k+1 = u_k + du to update

    For a linear residual this converges in one outer step.
    """

    rec: SolveRecorder

    def __init__(
        self,
        loss: DiscreteLoss,
        outer_maxiter: int = 1,
        outer_gtol: float = 1e-10,
        outer_ftol: float = 1e-12,
    ) -> None:
        super().__init__(loss)
        self.outer_gtol = outer_gtol
        self.outer_ftol = outer_ftol
        self.outer_maxiter = outer_maxiter

        # solve methods; "lu" is a direct sparse factorisation of A, the rest
        # are matrix-free iterative solves of A du = -r
        self.methods: Dict[str, Callable] = {
            "paradiag": spl.gmres,
            "gmres": spl.gmres,
        }

    def minimise(
        self,
        u0: Wavefield | np.ndarray | None = None,
        method: str = "paradiag",
        restart: int = 100,
        rtol: float = 1e-8,
        alpha: float | None = 0.001,
    ):
        """Minimise a DiscreteLoss using Gauss-Newton.

        Inner solves are orchestrated by spl.linalg iterative methods

        This implementation relies on the fact that for a nonsingular A,
        the exact minimiser of 1/2 ||A du + r||^2 is the solution to the
        linear problem A du = - r. This is the 'inner' problem we solve
        using an iterative method.

        This allows us to apply our block circulant preconditioner M that
        is an approximation of A-1.
        """

        grid = self.loss.problem.wavefield.grid
        nt, nx, ny = grid.nt, grid.nx, grid.ny
        N = nt * nx * ny
        s = self.loss.problem.sources[:, 0]  # single shot

        u0 = create_u0(u0, N)

        method = method.lower().strip()
        if method not in self.methods:
            raise ValueError(
                f"method must be one of {list(self.methods)}, got {method}"
            )

        if method == "gmres":
            alpha = None

        alpha = alpha if alpha else None

        meta = {
            "method": method,
            "alpha": alpha,
            "rtol": rtol,
            "restart": restart,
            "outer_ftol": self.outer_ftol,
            "outer_gtol": self.outer_gtol,
            "nt": nt,
            "nx": nx,
            "ny": ny,
        }
        self.rec = SolveRecorder(meta)

        result = self._minimise_gmres(u0, s, meta)

        return result

    def _minimise_gmres(
        self,
        u0: np.ndarray,
        s: np.ndarray,
        meta: Dict,
    ):
        alpha = meta["alpha"]
        rtol = meta["rtol"]
        restart = meta["restart"]

        we = self.loss.problem.wave_eq

        # extract reduced problem components
        B0, B1, B2 = we.reduced_blocks
        s, s0, s1 = we.reduced_rhs(s)
        Aop = CountedOperator(we.reduced_operator())

        self.rec.meta["norm_s"] = np.linalg.norm(s)

        # create preconditioner
        if alpha is not None:
            start = perf_counter()
            M = AlphaCirculantPreconditioner.from_wave_equation(we, alpha)
            M = M.as_linear_operator()
            end = perf_counter()
            self.rec.meta["t_setup"] = end - start
        else:
            M = None  # no precond for raw gmres
            restart = 3 * restart  # higher restart for raw
            self.rec.meta["t_setup"] = 0.0

        ns = we.nx * we.ny
        nt = we.nt
        ntm2 = nt - 2

        # gradient AT r
        def rmatvec(r: np.ndarray):
            R = r.reshape(ntm2, ns)
            G = R @ B0
            G[:-1] += R[1:] @ B1
            G[:-2] += R[2:] @ B2
            return G.ravel()

        u = (
            np.zeros(ntm2 * ns)
            if u0 is None
            else np.asarray(u0).reshape(nt, ns)[2:].ravel()
        )

        L_prev, nit, success = np.inf, 0, False
        message = f"Maximum outer iterations ({self.outer_maxiter}) reached"

        for nit in range(1, self.outer_maxiter + 1):
            r = Aop @ u - s  # reduced residual: THE r in A du = -r
            g = rmatvec(r)

            L = self.loss._eval_loss(r)
            self.rec.log(r, g)

            if np.linalg.norm(g) < self.outer_gtol:
                success, message = True, "Gradient norm below outer_gtol"
                break
            if abs(L_prev - L) < self.outer_ftol:
                success, message = True, "Loss change below outer_ftol"
                break
            L_prev = L

            hist = []
            c0 = Aop.count
            start = perf_counter()

            du, info = spl.gmres(
                Aop,
                -r,
                M=M,
                rtol=rtol,
                restart=restart,
                maxiter=1,
                callback=lambda pr: hist.append(pr),
                callback_type="pr_norm",
            )
            end = perf_counter()
            n_matvecs = Aop.count - c0

            inner = InnerRecord(
                residual_history=hist,
                true_relres=float(np.linalg.norm(Aop @ du + r))
                / self.rec.outers[-1].res,
                converged=(info == 0),
                n_matvecs=n_matvecs,
                t_solve=end - start,
            )

            self.rec.log_inner(inner)

            u = u + du  # alpha = 1 (exact for linear)

        U = np.empty((nt, ns))
        U[0], U[1], U[2:] = s0, s1, u.reshape(ntm2, ns)
        wf = Wavefield(we.wavefield.grid)
        wf.flat_data = U.ravel()
        return self.rec.finalise(wf, nit, success, message)

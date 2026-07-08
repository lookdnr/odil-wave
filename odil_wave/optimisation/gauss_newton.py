from .base import Optimiser
from odil_wave.wavefield import Wavefield
from odil_wave.loss import DiscreteLoss
from .utils import create_u0, OptimisationResult, InnerSolveInfo
from .precond import AlphaCirculantPreconditioner

from typing import Dict, Callable
import numpy as np
import scipy.sparse.linalg as spl


class GaussNewtonOptimiser(Optimiser):
    """Gauss Newton Optimiser. Approximates the Hessian by H ~ J^T J, where
    J is the Jacobian of the functional, then solves:

        - J^T J du = -J^t r for du (matrix-free, choice of solve method)
        - u_k+1 = u_k + du to update

    For a linear residual this converges in one outer step.
    """

    def __init__(
        self,
        loss: DiscreteLoss,
        outer_maxiter: int = 1,
        outer_gtol: float = 1e-8,
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
        N = grid.nt * grid.nx * grid.ny
        f = self.loss.problem.sources[:, 0]  # single shot

        u0 = create_u0(u0, N)

        method = method.lower().strip()
        if method not in self.methods:
            raise ValueError(
                f"method must be one of {list(self.methods)}, got {method}"
            )

        if method == "gmres":
            alpha = None

        result = self._minimise_gmres(u0, f, alpha if alpha else None, rtol, restart)

        return result

    def _minimise_gmres(
        self,
        u0: np.ndarray,
        f: np.ndarray,
        alpha: float | None = 0.001,
        rtol: float = 1e-8,
        restart: int = 10,
    ):
        we = self.loss.problem.wave_eq

        # extract reduced problem components
        B0, B1, B2 = we.reduced_blocks
        s, f0, f1 = we.reduced_rhs(f)
        Aop = we.reduced_operator()

        # create preconditioner
        if alpha is not None:
            M = AlphaCirculantPreconditioner.from_wave_equation(we, alpha)
            M = M.as_linear_operator()
        else:
            M = None  # no precond for raw gmres
            restart = 3 * restart  # higher restart for raw

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
        inner_history = []
        for nit in range(1, self.outer_maxiter + 1):
            r = Aop @ u - s  # reduced residual: THE r in A du = -r
            g = rmatvec(r)

            L = self.loss._eval_loss(r)
            self.loss.callback.log(L, r)

            if np.linalg.norm(g) < self.outer_gtol:
                success, message = True, "Gradient norm below outer_gtol"
                break
            if abs(L_prev - L) < self.outer_ftol:
                success, message = True, "Loss change below outer_ftol"
                break
            L_prev = L

            hist = []
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
            inner_history.append(
                InnerSolveInfo(
                    itn=len(hist),
                    normr=float(np.linalg.norm(Aop @ du + r)),
                    normar=None,
                    converged=(info == 0),
                )
            )

            u = u + du  # alpha = 1 (exact for linear)

        U = np.empty((nt, ns))
        U[0], U[1], U[2:] = f0, f1, u.reshape(ntm2, ns)
        wf = Wavefield(we.wavefield.grid)
        wf.flat_data = U.ravel()
        return OptimisationResult(
            wf,
            self.loss.callback,
            nit=nit,
            success=success,
            message=message,
            inner_history=inner_history,
        )

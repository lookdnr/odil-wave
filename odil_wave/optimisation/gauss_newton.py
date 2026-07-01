from .base import Optimiser
from odil_wave.wavefield import Wavefield
from odil_wave.loss import DiscreteLoss
from .utils import create_u0, OptimisationResult, InnerSolveInfo

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
        outer_maxiter: int = 20,
        outer_gtol: float = 1e-8,
        outer_ftol: float = 1e-12,
    ) -> None:
        super().__init__(loss)
        self.outer_gtol = outer_gtol
        self.outer_ftol = outer_ftol
        self.outer_maxiter = outer_maxiter

        # methods for solving normal equation
        self.methods: Dict[str, Callable] = {
            "gmres": spl.gmres,
            "lsmr": spl.lsmr,
            "lsqr": spl.lsqr,
        }

    def minimise(
        self,
        u0: Wavefield | np.ndarray | None = None,
        method: str = "gmres",
        inner_maxiter: int = 20,
        atol: float = 1e-8,
        btol: float = 1e-8,
        rtol: float = 1e-8,
        callback=None,
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
        we = self.loss.problem.wave_eq
        grid = self.loss.problem.wavefield.grid
        N = grid.nt * grid.nx * grid.ny
        f = self.loss.problem.sources[:, 0]  # single shot

        u0 = create_u0(u0, N)

        method = method.lower().strip()
        if method not in self.methods:
            raise ValueError(
                f"method must be one of {list(self.methods)}, got {method}"
            )

        # create A operator and assign operations
        Aop = spl.LinearOperator(
            shape=(N, N),
            matvec=we.matvec,  # type: ignore
            rmatvec=we.rmatvec,  # type: ignore
            dtype=np.float64,
        )

        L_prev = np.inf
        u = u0
        success = False
        message = f"Maximum outer iterations ({self.outer_maxiter}) reached"
        nit = 0
        inner_history: list[InnerSolveInfo] = []
        for nit in range(1, self.outer_maxiter + 1):
            # compute residual and gradient
            r = we.residual(u, f)  # compute residual vector
            g = we.rmatvec(r)  # J^T r (grad)

            # compute loss and log
            L = self.loss._eval_loss(r)
            self.loss.callback.log(L, r)

            if np.linalg.norm(g) < self.outer_gtol:
                success, message = True, "Gradient norm below outer_gtol"
                break
            if abs(L_prev - L) < self.outer_ftol:
                success, message = True, "Loss change below outer_ftol"
                break
            L_prev = L

            # gmres and lsmr/lsqr expose different controls, so branch methods here

            if method == "gmres":
                du, info = spl.gmres(Aop, -r, M=None, rtol=rtol, maxiter=inner_maxiter)
                inner_history.append(
                    InnerSolveInfo(
                        itn=info if info > 0 else inner_maxiter,
                        normr=None,
                        normar=None,
                        converged=(info == 0),
                    )
                )

            elif method == "lsmr":  # lsmr / lsqr: min ‖A·du + r‖²
                du, istop, itn, normr, normar, *_ = spl.lsmr(
                    Aop, -r, atol=atol, btol=btol, maxiter=inner_maxiter
                )
                inner_history.append(
                    InnerSolveInfo(
                        itn=itn, normr=normr, normar=normar, converged=(istop != 7)
                    )
                )

            else:
                du, istop, itn, r1norm, _, _, _, arnorm, *_ = spl.lsqr(
                    Aop, -r, atol=atol, btol=btol, iter_lim=inner_maxiter
                )
                inner_history.append(
                    InnerSolveInfo(
                        itn=itn, normr=r1norm, normar=arnorm, converged=(istop != 7)
                    )
                )

            # update
            u = u + du  # alpha = 1 (exact for linear)

            if callback is not None:
                callback(u)

        wf = Wavefield(self.loss.problem.wave_eq.wavefield.grid)
        wf.flat_data = u
        return OptimisationResult(
            wf,
            self.loss.callback,
            nit=nit,
            success=success,
            message=message,
            inner_history=inner_history,
        )

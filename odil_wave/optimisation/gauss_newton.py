from .base import Optimiser
from odil_wave.wavefield import Wavefield
from odil_wave.loss import DiscreteLoss
from .utils import create_u0

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
            "lsmr": spl.lsmr,
            "lsqr": spl.lsqr,
            "cg": spl.cg,
        }

    def minimise(
        self,
        u0: Wavefield | np.ndarray | None = None,
        method: str = "lsmr",
        inner_maxiter: int = 20,
        atol: float = 1e-8,
        btol: float = 1e-8,
        rtol: float = 1e-8,
        callback=None,
    ):
        """Minimise a DiscreteLoss using Gauss-Newton.

        Inner solves are orchestrated by spl.linalg iterative methods
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

        # lambdas for spl.LinearOperator
        matvec = lambda x: we.matvec(x)  # noqa
        rmatvec = lambda x: we.rmatvec(x)  # noqa

        # create A operator and assign operations
        Aop = spl.LinearOperator(np.float64, (N, N))
        Aop.matvec = matvec  # , rmatvec=rmatvec, dtype=np.float64)
        Aop.rmatvec = rmatvec

        # normal-equations operator for cg: Hv ~ JᵀJ v
        # lambda below computes JTJ (x)
        matvec = lambda x: we.rmatvec(we.matvec(x))  # noqa
        H = spl.LinearOperator(np.float64, (N, N))
        H.matvec = matvec

        L_prev = np.inf
        u = u0
        for _ in range(self.outer_maxiter):
            # compute residual and gradient
            r = we.residual(u, f)  # compute residual vector
            g = we.rmatvec(r)  # J^T r (grad)

            # compute loss and log
            L = self.loss._eval_loss(r)
            self.loss.callback.log(L, r)

            if np.linalg.norm(g) < self.outer_gtol or abs(L_prev - L) < self.outer_ftol:
                break
            L_prev = L

            # cg and lsmr/lsqr expose different controls, so branch methods here
            if method == "cg":
                du, _ = spl.cg(
                    H, -g, rtol=rtol, maxiter=inner_maxiter
                )  # different callsite

            else:  # lsmr / lsqr: min ‖A·du + r‖²
                du = self.methods[method](
                    Aop, -r, atol=atol, btol=btol, maxiter=inner_maxiter
                )[0]

            # update
            u = u + du  # alpha = 1 (exact for linear)

            if callback is not None:
                callback(u)

        # return Wavefield object
        wf = Wavefield(grid=grid)
        wf.flat_data = u
        return wf, self.loss.callback

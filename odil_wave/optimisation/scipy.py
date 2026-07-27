import scipy.optimize as scopt
import numpy as np

from .base import Optimiser
from odil_wave.loss import DiscreteLoss
from odil_wave.wavefield import Wavefield
from odil_wave.metrics import SolveRecorder, SolveResult

from .utils import create_u0


class ScipyOptimiser(Optimiser):
    """Wrapper around scipy.optimize.minimize"""

    def __init__(self, loss: DiscreteLoss, method: str = "L-BFGS-B", **opts) -> None:
        super().__init__(loss)
        self.method = method  # e.g., 'L-BFGS-B', 'Newton-CG'
        self.opts = opts  # e.g., maxiter, ftol

    def minimise(
        self,
        u0: Wavefield | np.ndarray | None = None,
        maxiter=500,
        ftol=1e-8,
        gtol=1e-10,
        callback=None,
    ) -> SolveResult:

        self.opts.update(maxiter=maxiter, ftol=ftol, gtol=gtol)

        grid = self.loss.problem.wavefield.grid
        nt, nx, ny = grid.nt, grid.nx, grid.ny
        N = nt * nx * ny
        s = self.loss.problem.sources[:, 0]

        meta = {
            "method": self.method,
            "maxiter": maxiter,
            "ftol": ftol,
            "gtol": gtol,
            "nt": nt,
            "nx": nx,
            "ny": ny,
            "norm_s": np.linalg.norm(s),
        }
        rec = SolveRecorder(meta)
        self.loss.callback = rec

        u0 = create_u0(u0, N)

        result = scopt.minimize(
            fun=self.loss.evaluate,
            x0=u0,
            method=self.method,
            jac=True,  # analytic gradient via rmatvec in ForwardLoss._grad
            callback=callback,
            options=self.opts,
        )

        if not result.success:
            print(f"Warning: optimisation did not converge: {result.message}")

        wf = Wavefield(grid=grid)
        wf.flat_data = result.x
        return rec.finalise(
            wf, nit=result.nit, success=result.success, message=str(result.message)
        )


class LBFGSB(ScipyOptimiser):
    """Subclass for L-BFGS-B"""

    def __init__(self, loss: DiscreteLoss, **opts) -> None:
        super().__init__(loss, method="L-BFGS-B", **opts)

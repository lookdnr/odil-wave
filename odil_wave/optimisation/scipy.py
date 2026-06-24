from typing import Tuple

import scipy.optimize as scopt
import numpy as np

from .base import Optimiser
from odil_wave.loss import DiscreteLoss
from odil_wave.wavefield import Wavefield
from odil_wave.loss.utils import LossTape


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
    ) -> Tuple[Wavefield, LossTape]:

        self.opts.update(maxiter=maxiter, ftol=ftol, gtol=gtol)

        grid = self.loss.problem.wavefield.grid
        N = grid.nt * grid.nx * grid.ny

        if u0 is None:
            u0 = np.zeros(N)
        elif isinstance(u0, Wavefield):
            u0 = u0.flat_data
        elif isinstance(u0, np.ndarray):
            u0 = np.asarray(u0).ravel()
        else:
            raise TypeError(
                "arg `u0` must be one of Wavefield, np.ndarray, None, got"
                + f" {type(u0)}"
            )

        result = scopt.minimize(
            fun=self.loss.evaluate,
            x0=u0,
            method=self.method,
            jac=True,  # analytic gradient via rmatvec in ForwardLoss._grad
            callback=callback,
            options=self.opts,
        )
        self.loss.callback.result = result

        if not result.success:
            print(f"Warning: optimisation did not converge: {result.message}")

        wf = Wavefield(grid=grid)
        wf.flat_data = result.x
        return wf, self.loss.callback


class LBFGSB(ScipyOptimiser):
    """Subclass for L-BFGS-B"""

    def __init__(self, loss: DiscreteLoss, **opts) -> None:
        super().__init__(loss, method="L-BFGS-B", **opts)

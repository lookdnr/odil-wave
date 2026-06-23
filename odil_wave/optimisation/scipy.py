from typing import Tuple, List

import scipy.optimize as scopt
import numpy as np

from .base import Optimiser
from odil_wave.loss import DiscreteLoss, ForwardLoss
from odil_wave.wavefield import Wavefield
from odil_wave.loss.utils import LossTape


class ScipyOptimiser(Optimiser):
    """Wrapper around scipy.optimize.minimize"""

    def __init__(
        self, wavefield: Wavefield, loss: DiscreteLoss, method: str = "L-BFGS-B", **opts
    ) -> None:
        super().__init__(wavefield, loss)
        self.method = method  # e.g., 'L-BFGS-B', 'Newton-CG'
        self.opts = opts  # e.g., maxiter, ftol

    def minimise(
        self, maxiter=500, ftol=1e-8, gtol=1e-10, callback=None
    ) -> Tuple[List[Wavefield], LossTape]:

        # apply specified config
        self.opts["maxiter"] = maxiter
        self.opts["ftol"] = ftol
        self.opts["gtol"] = gtol

        # use amplitude data only for the forward
        if isinstance(self.loss, ForwardLoss):
            # tile amplitude for each shot, since forward loss only optimises amplitude
            u0 = np.tile(
                self.wavefield.amplitude.cpu().numpy().ravel(),
                self.loss.config.geometry.n_sources,
            )

        result = scopt.minimize(
            fun=self.loss.evaluate,
            x0=u0,
            method=self.method,
            jac=True,  # gradient provided by torch through .evaluate
            callback=callback,
            options=self.opts,
        )
        self.loss.callback.result = result  # store optimisation result in loss callback

        if not result.success:
            print(f"Warning: Optimisation did not converge: {result.message}")

        # cast result into list of per-shot wavefields
        outputs = []
        if isinstance(self.loss, ForwardLoss):
            n_shots = self.loss.config.geometry.n_sources
            chunks = result.x.reshape(n_shots, -1)  # (n_shots, Nt*Nx*Ny)

            for s in range(n_shots):
                wf = Wavefield(
                    grid=self.wavefield.grid, init_wavespeed=self.wavefield.wavespeed
                )
                wf.amplitude = chunks[s]
                outputs.append(wf)

        return outputs, self.loss.callback


class LBFGSB(ScipyOptimiser):
    """Subclass for L-BFGS method"""

    def __init__(
        self,
        wavefield: Wavefield,
        loss: DiscreteLoss,
        maxiter: int = 500,
        ftol: float = 1e-8,
        gtol: float = 1e-10,
        **opts,
    ) -> None:
        super().__init__(
            wavefield,
            loss,
            method="L-BFGS-B",
            maxiter=maxiter,
            ftol=ftol,
            gtol=gtol,
            **opts,
        )

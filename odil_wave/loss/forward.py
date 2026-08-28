from .base import DiscreteLoss
from typing import Tuple

import numpy as np


class ForwardLoss(DiscreteLoss):
    """Loss function for the forward ODIL problem.

    Encapsulates ||Au - s||^2, the discrete PDE residual norm, over the
    wavefield `u` for a fixed wavespeed and source term.

    Notes
    -----
    See also `DiscreteLoss` which `ForwardLoss` is derived from.
    """

    def _residuals(self, u: np.ndarray, sources: np.ndarray) -> np.ndarray:
        """np.ndarray: compute the PDE residual Au - s for a candidate wavefield u."""
        r_pde = self.problem.wave_eq.residual(u, sources)
        return r_pde

    def _eval_loss(self, residuals: np.ndarray) -> np.float64:
        """np.float64: evaluate the sum of squared residuals."""
        return np.sum(residuals**2)

    def _grad(self, r: np.ndarray) -> np.ndarray:
        """Analytical gradient for the wave equation
            L = ||Au - f||^2 = Sum_i r_i^2,
            so dL/du = 2A^T r by the chain rule

        This form is required for `scipy.optimize.minimize`.
        """
        return 2 * self.problem.wave_eq.rmatvec(r)

    def evaluate(self, u: np.ndarray) -> Tuple[np.float64, np.ndarray]:
        """Evalutes the forward loss and gradient for a candidate wavefield `u`.

        Parameters
        ----------
        u : np.ndarray
            The candidate wavefield (current iterate).

        Returns
        -------
        L : float
            Scalar loss value, ||Au - s||^2.
        g : np.ndarray
            Gradient of the loss with respect to `u`

        Notes
        -----
        Only the first shot's source term is used. Multishot support is pending.
        """

        #  get wavespeed separately, we need it to compute the PDE residuals
        sources = self.problem.sources[:, 0]  # TODO single shot for now

        residual = self._residuals(u, sources)
        L = self._eval_loss(residual)

        self.evaluations += 1

        g = self._grad(residual)
        if self.do_logging and self.evaluations % self.log_every == 0:
            self.callback.log(residual, g)

        return L, g

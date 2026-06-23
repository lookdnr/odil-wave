from .base import DiscreteLoss
from odil_wave.wavefield import Wavefield

import numpy as np


class ForwardLoss(DiscreteLoss):
    """Loss function for the forward problem."""

    def _residuals(self, wavefield: Wavefield, sources: np.ndarray) -> np.ndarray:
        r_pde = self.problem.wave_eq.residual(wavefield, sources)
        return r_pde

    def _eval_loss(self, residuals: np.ndarray) -> np.float64:
        return np.sum(residuals**2)

    def evaluate(self, wavefield: Wavefield) -> np.float64:

        #  get wavespeed separately, we need it to compute the PDE residuals
        sources = self.problem.sources

        residual = self._residuals(wavefield, sources)
        L = self._eval_loss(residual)

        self.evaluations += 1

        if self.evaluations % self.callback.log_every == 0:
            self.callback.log(L, residual)

        return L

from abc import ABC, abstractmethod

from odil_wave.loss import DiscreteLoss
from odil_wave.wavefield import Wavefield
from odil_wave.metrics import SolveResult

import numpy as np


class Optimiser(ABC):
    """Base class for wavefield optimisers.

    Parameters
    ----------
    loss : DiscreteLoss
        Loss function (and underlying `Problem`) being minimised.
    """

    def __init__(self, loss: DiscreteLoss) -> None:
        self.loss = loss  # loss function

    @abstractmethod  # to be implemented by classes that inherit
    def minimise(
        self,
        u0: Wavefield | np.ndarray | None = None,
        maxiter: int = 500,
        ftol: float = 1e-8,
        gtol: float = 1e-10,
    ) -> SolveResult:
        """Minimise the loss and return the solution plus solve history.

        Parameters
        ----------
        u0 : Wavefield or np.ndarray, optional
            Initial guess; zero-initialised if None.
        maxiter : int, optional
            Maximum number of iterations.
        ftol : float, optional
            Loss change convergence tolerance.
        gtol : float, optional
            Gradient norm convergence tolerance.

        Returns
        -------
        SolveResult
            Solution wavefield and recorded solve history.
        """
        pass

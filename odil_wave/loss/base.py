from abc import ABC, abstractmethod
from typing import Tuple
from odil_wave.metrics.recording import SolveRecorder
from odil_wave.utils import Problem
import numpy as np


class DiscreteLoss(ABC):
    """Base class for discrete loss functions.

    Wraps an optimisation `Problem` (bundling grid, model, geometry, and PDE
    operators) together with an optional `SolveRecorder` callback, and defines
    residual/ gradient interfaces that `ForwardLoss` implements.

    Parameters
    ----------
    problem : Problem
        Optimisation problem configuration (grid, model, geometry, operators)
        the loss is evaluated against.
    callback : SolveRecorder
        Recoder invoked with residual and gradient every `log_every` evaluations,
        a fresh instance is instantiated if None.
    log_every : int
        Log to `callback` every `log_every` evaluations. Logging is disabled
        if 0.

    Attributes
    ----------
    evaluations : int
        Running count of calls to `evaluate`

    Raises
    ------
    ValueError
        If `log_every` is negative
    """

    def __init__(
        self,
        problem: Problem,
        callback: SolveRecorder | None = None,
        log_every: int = 1,
    ):
        self.problem = problem  # loss configuration
        self.callback = (
            callback if callback is not None else SolveRecorder()
        )  # loss history callback

        self.do_logging = True
        if log_every < 0:
            raise ValueError(f"arg log_every must be >= 0, got {log_every}")
        if log_every == 0:
            self.do_logging = False

        self.log_every = log_every
        self.evaluations = 0  # counter for number of loss evaluations

    @abstractmethod
    def evaluate(self, wavefield: np.ndarray) -> Tuple[float, np.ndarray]:
        """Evaluate the loss and its gradient for a given wavefield.

        Parameters
        ----------
        wavefield : np.ndarray
            Candidate wavefield (current iterate) to evaluate the loss on.

        Returns
        -------
        loss : float
            Scalar loss value.
        grad : np.ndarray
            Gradient of the loss with respect to `wavefield`.
        """
        pass

    @abstractmethod
    def _eval_loss(self, residuals: np.ndarray) -> np.float64:
        """np.float64: scalar loss. Implemented by `ForwardLoss`"""
        pass

    @abstractmethod
    def _residuals(self, wavefield: np.ndarray) -> np.ndarray:
        """np.ndarray: residual vector v = Au - s. Implemented by `ForwardLoss`"""
        pass

    @abstractmethod
    def _grad(self, r: np.ndarray) -> np.ndarray:
        """np.ndarray: the gradient evaluted on the residual vector `r`.
        Implemented by `ForwardLoss`"""
        pass

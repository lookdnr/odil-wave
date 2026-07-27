from abc import ABC, abstractmethod
from typing import Tuple
from odil_wave.metrics.recording import SolveRecorder
from odil_wave.utils import Problem
import numpy as np


class DiscreteLoss(ABC):
    """Base class for discrete loss functions."""

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
        """Evaluate the loss function given a wavefield."""
        pass

    @abstractmethod
    def _eval_loss(self, residuals: np.ndarray) -> np.float64:
        pass

    @abstractmethod
    def _residuals(self, wavefield: np.ndarray) -> np.ndarray:
        """Compute the residuals of the loss function given a wavefield."""
        pass

    @abstractmethod
    def _grad(self, r: np.ndarray) -> np.ndarray:
        pass

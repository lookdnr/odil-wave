from abc import ABC, abstractmethod
from typing import Tuple
from .utils import LossTape
from odil_wave.utils import Problem
import numpy as np

from odil_wave.wavefield import Wavefield


class DiscreteLoss(ABC):
    """Base class for discrete loss functions."""

    def __init__(
        self,
        problem: Problem,
        callback: LossTape | None = None,
    ):
        self.problem = problem  # loss configuration
        self.callback = (
            callback if callback is not None else LossTape()
        )  # loss history callback

        self.evaluations = 0  # counter for number of loss evaluations

    @abstractmethod
    def evaluate(self, wavefield: Wavefield) -> Tuple[float, np.ndarray]:
        """Evaluate the loss function given a wavefield."""
        pass

    @abstractmethod
    def _eval_loss(self, residuals: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _residuals(self, wavefield: Wavefield) -> np.ndarray:
        """Compute the residuals of the loss function given a wavefield."""
        pass

    @abstractmethod
    def _grad(self, r: np.ndarray) -> np.ndarray:
        pass

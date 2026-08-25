from abc import ABC, abstractmethod

from odil_wave.loss import DiscreteLoss
from odil_wave.wavefield import Wavefield
from odil_wave.metrics import SolveResult

import numpy as np


class Optimiser(ABC):
    """Base optimiser class"""

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
        pass

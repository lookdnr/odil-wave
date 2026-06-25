from abc import ABC, abstractmethod

from odil_wave.loss import DiscreteLoss
from odil_wave.wavefield import Wavefield


class Optimiser(ABC):
    """Base optimiser class"""

    def __init__(self, loss: DiscreteLoss) -> None:
        self.loss = loss  # loss function

    @abstractmethod  # to be implemented by classes that inherit
    def minimise(self, u0: Wavefield, maxiter: int, ftol: float, gtol: float):
        pass

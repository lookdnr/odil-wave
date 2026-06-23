from abc import ABC, abstractmethod

from odil_wave.loss import DiscreteLoss
from odil_wave.wavefield import Wavefield


class Optimiser(ABC):
    """Base optimiser class"""

    def __init__(self, wavefield: Wavefield, loss: DiscreteLoss) -> None:
        self.loss = loss  # loss function
        self.wavefield = wavefield  # wavefield to optimise

    @abstractmethod  # to be implemented by classes that inherit
    def minimise(self, maxiter, ftol, gtol, callback=None):
        pass

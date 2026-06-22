from abc import ABC, abstractmethod
import scipy.sparse as sp
from odil_wave.wavefield import Wavefield


class SparseOperator(ABC):
    """Base class for sparse operators used in direct solves"""

    def __init__(self, wavefield: Wavefield) -> None:
        self.wavefield = wavefield

    @abstractmethod
    def assemble(self) -> None:
        pass

    @abstractmethod
    def apply(self, wavefield) -> sp.spmatrix:
        pass

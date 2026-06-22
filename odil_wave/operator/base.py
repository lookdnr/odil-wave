from abc import ABC, abstractmethod
import scipy.sparse as sp
from odil_wave.wavefield import Wavefield


class SparseOperator(ABC):
    """Base class for sparse operators used in direct solves"""

    def __init__(self, wavefield: Wavefield, ord: int = 2) -> None:
        self.wavefield = wavefield
        self.ord = ord

    @abstractmethod
    def assemble(self) -> sp.csr_matrix:
        pass

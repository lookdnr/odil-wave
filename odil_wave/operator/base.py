from abc import ABC, abstractmethod
import numpy as np
from odil_wave.wavefield import Wavefield


class SparseOperator(ABC):
    """Base class for sparse operators used in direct solves.

    Parameters
    ----------
    wavefield : Wavefield
        Wavefield the operator acts on, supplies the grid it's
        discretised on.
    ord : {2, 4, 6, 8}
        Finite difference accuracy order.

    Raises
    ------
    ValueError
        If `ord` is not one of 2, 4, 6, 8.
    """

    def __init__(self, wavefield: Wavefield, ord: int = 2) -> None:
        self.wavefield = wavefield

        if ord not in [2, 4, 6, 8]:
            raise ValueError("accuracy order 'ord' must be one of 2, 4, 6, 8")

        self.ord = ord

    @abstractmethod
    def assemble(self):
        """Assemble the sparse matrix operator.

        Returns
        -------
        scipy.sparse.spmatrix
            The assembled operator matrix.
        """
        pass

    @abstractmethod
    def apply(self, U: np.ndarray) -> np.ndarray:
        """Apply the operator to a (time, space) array.

        Parameters
        ----------
        U : np.ndarray
            (nt, nx*ny) array to apply the operator to.

        Returns
        -------
        np.ndarray
            Result of applying the operator to `U`.
        """
        pass

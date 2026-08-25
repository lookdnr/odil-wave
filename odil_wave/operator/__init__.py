"""Discrete spatial/ temporal derivative operators and the assembled wave equation.

`Laplacian`, `FirstTimeDerivative`, `SecondTimeDerivative` are the sparse
building blocks.

`WaveEquation` assembles them (with Higdon absorbing boundary conditions)
into the full discrete PDE operator A used for the forward ODIL solve.
"""

from .spatial import Laplacian
from .temporal import FirstTimeDerivative, SecondTimeDerivative
from .wave import WaveEquation

__all__ = [
    "Laplacian",
    "FirstTimeDerivative",
    "SecondTimeDerivative",
    "WaveEquation",
]

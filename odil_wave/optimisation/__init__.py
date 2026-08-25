"""Optimisers for the ODIL wave equation solve.

`GaussNewtonOptimiser` drives the matrix free Gauss-Newton solve
with an optional ParaDiag (`AlphaCirculantPreconditioner`) preconditioner.
`LBFGSB` wraps `scipy.optimize.minimize` for gradient based baselines.
"""

from .scipy import LBFGSB
from .gauss_newton import GaussNewtonOptimiser

__all__ = ["LBFGSB", "GaussNewtonOptimiser"]

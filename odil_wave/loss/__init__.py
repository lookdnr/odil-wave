"""Loss functions for the ODIL solve.

`DiscreteLoss` is a base class defining the interface. `ForwardLoss`
implements it for the fixed wavespeed forward problem.
"""

from .base import DiscreteLoss
from .forward import ForwardLoss

__all__ = ["DiscreteLoss", "ForwardLoss"]

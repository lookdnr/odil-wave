"""Wavefield representation for the wave solver.

Provides `Wavefield`, the (nt, nx*ny) amplitude field that the ODIL
forward solve optimises over.
"""

from .base import Wavefield

__all__ = ["Wavefield"]

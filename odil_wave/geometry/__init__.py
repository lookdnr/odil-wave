"""Acqusition geometry: source and receiever configuration for a simulation."""

from .acquisition_geometry import AcquisitionGeometry
from .sources import Sources
from .receivers import Receivers

__all__ = ["AcquisitionGeometry", "Sources", "Receivers"]

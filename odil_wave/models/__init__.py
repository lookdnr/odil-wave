"""Velocity models a wave equation problem.

Provides `VelocityModel` (the abstract base) and subclasses:

- `HomogeneousModel`
- `SheppLoganModel`
- `OverDensityModel`
- `CustomModel`
"""

from .velocity_models import (
    SheppLoganModel,
    HomogeneousModel,
    OverDensityModel,
    CustomModel,
)

__all__ = ["SheppLoganModel", "HomogeneousModel", "OverDensityModel", "CustomModel"]

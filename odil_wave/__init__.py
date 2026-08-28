"""odil_wave: solving the 2D wave equation via ODIL.

Provides the full pipeline for an ODIL solve:

grid and velocity model setup
(`Grid`, `SheppLoganModel`, `HomogeneousModel`, `OverDensityModel`),

acquisition geometry (`Sources`, `Receivers`, `AcquisitionGeometry`),

the discrete wave equation operator (`WaveEquation`),

wavefield (`Wavefield`),

a `Problem`/`ForwardLoss` pair defining the Gauss-Newton least-squares problem,

matrix free optimisers (`GaussNewtonOptimiser`, with a ParaDiag preconditioner)

`SolveRecorder`/`SolveResult` for recording and inspecting convergence history.
"""

from odil_wave.grid import Grid
from odil_wave.geometry import AcquisitionGeometry, Sources, Receivers
from odil_wave.models import SheppLoganModel, HomogeneousModel, OverDensityModel
from odil_wave.wavefield import Wavefield
from odil_wave.operator import WaveEquation
from odil_wave.utils import Problem
from odil_wave.loss import ForwardLoss
from odil_wave.metrics import SolveRecorder, SolveResult
from odil_wave.optimisation import LBFGSB, GaussNewtonOptimiser

__all__ = [
    "Grid",
    "AcquisitionGeometry",
    "Sources",
    "Receivers",
    "SheppLoganModel",
    "HomogeneousModel",
    "OverDensityModel",
    "Wavefield",
    "WaveEquation",
    "Problem",
    "SolveRecorder",
    "SolveResult",
    "ForwardLoss",
    "LBFGSB",
    "GaussNewtonOptimiser",
]

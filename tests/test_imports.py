"""Import smoke tests."""

import importlib

import odil_wave

PUBLIC_API = [
    "Grid",
    "AcquisitionGeometry",
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
]

SUBMODULES = [
    "odil_wave.grid.grid",
    "odil_wave.geometry.acquisition_geometry",
    "odil_wave.geometry.sources",
    "odil_wave.geometry.receivers",
    "odil_wave.geometry.utils",
    "odil_wave.models.base",
    "odil_wave.models.velocity_models",
    "odil_wave.wavefield.base",
    "odil_wave.operator.base",
    "odil_wave.operator.boundaries",
    "odil_wave.operator.ghost",
    "odil_wave.operator.spatial",
    "odil_wave.operator.stencils",
    "odil_wave.operator.temporal",
    "odil_wave.operator.wave",
    "odil_wave.loss.base",
    "odil_wave.loss.forward",
    "odil_wave.metrics.recording",
    "odil_wave.optimisation.base",
    "odil_wave.optimisation.scipy",
    "odil_wave.optimisation.gauss_newton",
    "odil_wave.optimisation.utils",
    "odil_wave.utils.problem",
]


def test_public_api_exported():
    for name in PUBLIC_API:
        assert hasattr(odil_wave, name), f"missing public export: {name}"
        assert name in odil_wave.__all__


def test_all_submodules_import():
    for mod in SUBMODULES:
        importlib.import_module(mod)

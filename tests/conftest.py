import matplotlib

matplotlib.use("Agg")  # headless

import pytest

from odil_wave import (
    Grid,
    HomogeneousModel,
    Wavefield,
    WaveEquation,
    AcquisitionGeometry,
    Problem,
    ForwardLoss,
)
from odil_wave.geometry import Sources, Receivers

"""Shared fixtures for smoke tests"""


@pytest.fixture
def grid():
    return Grid(nx=11, ny=13, xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0, t_max=0.4)


@pytest.fixture
def model(grid):
    return HomogeneousModel(grid, background_c=1.5)


@pytest.fixture
def wavefield(grid):
    return Wavefield(grid)


@pytest.fixture
def sources(grid):
    return Sources(grid, n_sources=1, mode="custom", source_locs=((0.0, 0.0),))


@pytest.fixture
def receivers(grid):
    return Receivers(grid, mode="ring", n_receivers=8)


@pytest.fixture
def geometry(sources, receivers):
    return AcquisitionGeometry(sources, receivers)


@pytest.fixture
def wave_eq(wavefield, model):
    return WaveEquation(wavefield, model)


@pytest.fixture
def problem(wave_eq, geometry):
    return Problem(wave_eq, geometry)


@pytest.fixture
def loss(problem):
    return ForwardLoss(problem)

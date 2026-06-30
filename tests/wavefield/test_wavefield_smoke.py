import numpy as np

from odil_wave import Wavefield


def test_wavefield_instantiates(wavefield, grid):
    assert wavefield.U.shape == (grid.nt, grid.nx * grid.ny)
    assert wavefield.flat_data.shape == (grid.nt * grid.nx * grid.ny,)


def test_wavefield_with_init_amplitude(grid):
    amp = np.zeros((grid.nt, grid.nx * grid.ny))
    wf = Wavefield(grid, init_amplitude=amp)
    assert wf.U.shape == (grid.nt, grid.nx * grid.ny)

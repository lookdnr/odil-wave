import numpy as np


def test_wave_equation_instantiates(wave_eq, grid):
    N = grid.nt * grid.nx * grid.ny
    u = np.zeros(N)
    assert wave_eq.matvec(u).shape == (N,)
    assert wave_eq.rmatvec(u).shape == (N,)
    assert wave_eq.residual(u, u).shape == (N,)

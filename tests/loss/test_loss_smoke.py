import numpy as np

from odil_wave import SolveRecorder


def test_problem_instantiates(problem, grid):
    ns = grid.nx * grid.ny
    assert problem.sources.shape[0] == grid.nt * ns


def test_forward_loss_evaluates(loss, grid):
    N = grid.nt * grid.nx * grid.ny
    L, g = loss.evaluate(np.zeros(N))
    assert np.isscalar(L) or np.ndim(L) == 0
    assert g.shape == (N,)


def test_loss_tape_instantiates():
    rec = SolveRecorder()
    rec.log(np.zeros(3))
    assert rec.outers[0].res == 0.0

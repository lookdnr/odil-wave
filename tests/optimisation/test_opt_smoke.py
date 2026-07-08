from odil_wave import LBFGSB, Wavefield
from odil_wave.loss.utils import LossTape
from odil_wave.optimisation import GaussNewtonOptimiser


def test_lbfgsb_instantiates_and_runs(loss):
    opt = LBFGSB(loss)
    wf, tape = opt.minimise(maxiter=2)
    assert isinstance(wf, Wavefield)
    assert isinstance(tape, LossTape)


def test_gauss_newton_instantiates_and_runs(loss):
    opt = GaussNewtonOptimiser(loss, outer_maxiter=2)
    result = opt.minimise(method="paradiag")
    assert isinstance(result.solution, Wavefield)

    opt = GaussNewtonOptimiser(loss, outer_maxiter=2)
    result = opt.minimise(method="gmres")
    assert isinstance(result.solution, Wavefield)

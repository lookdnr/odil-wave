import numpy as np
import pytest
from odil_wave.operator import FirstTimeDerivative, SecondTimeDerivative, Laplacian
from odil_wave import Grid, Wavefield


# utility to obtain observed order of convergence
def observed_order(hs, errs):
    """Slope of log(err) vs log(h), the empirical convergence rate"""
    hs, errs = np.asarray(hs), np.asarray(errs)
    return np.polyfit(np.log(hs), np.log(errs), 1)[0]  # order 1 polynomial


# ===== Order of convergence for temporal operators =====


def _time_error(nt, ord, deriv):
    grid = Grid(interior_shape=(20, 30), pml_width=0, init_nt=nt, t_max=1.0)
    wf = Wavefield(grid)
    t = grid.t
    w = 2 * np.pi

    # create field with analytical solution
    wf.U = np.broadcast_to(np.sin(w * t)[:, None], wf.U.shape).copy()

    if deriv == 1:
        out = FirstTimeDerivative(wf, ord=ord).apply(wf.U)[:, 0]
        exact = w * np.cos(w * t)  # exact deriavtive
    else:
        out = SecondTimeDerivative(wf, ord=ord).apply(wf.U)[:, 0]
        exact = -(w**2) * np.sin(w * t)

    half = ord // 2  # interior

    # compute error as max absolute diff
    err = np.max(np.abs(out[half:-half] - exact[half:-half]))
    return grid.dt, err


@pytest.mark.parametrize("deriv", [1, 2])
@pytest.mark.parametrize("ord", [2, 4, 6])  # 8 converges too quickly
def test_temporal_convergence(ord, deriv):
    nts = [41, 61, 91, 131]  # refine init nt
    hs, errs = zip(*(_time_error(nt, ord, deriv) for nt in nts))
    obs = observed_order(hs, errs)
    print(f"\n[TEMPORAL] deriv: {deriv}, order: {ord}, observed order: {obs}")
    assert np.allclose(obs, ord, rtol=1e-1)


# ===== Order of convergence for Laplacian operator =====


def _space_error(n, ord):
    grid = Grid(interior_shape=(n, n + 4), pml_width=0)
    wf = Wavefield(grid)
    X, Y = grid.X, grid.Y
    a, b = np.pi, np.pi

    # create field with analytical soln
    # lap(u) = -a^2 d^2u/dx^2 - b^2 d^2u/dy^2 => -(a^2 + b^2)u
    u = np.sin(a * X) * np.sin(b * Y)
    wf.U = np.broadcast_to(u.ravel(), wf.U.shape).copy()

    out = Laplacian(wf, ord=ord).apply(wf.U)[0].reshape(grid.nx, grid.ny)
    exact = -(a**2 + b**2) * u

    half = ord // 2
    err = np.max(np.abs(out[half:-half, half:-half] - exact[half:-half, half:-half]))
    return grid.dx, err


@pytest.mark.parametrize("ord", [2, 4, 6])
def test_laplacian_convergence(ord):
    ns = [41, 61, 91, 131]
    hs, errs = zip(*(_space_error(n, ord) for n in ns))
    obs = observed_order(hs, errs)
    print(f"\n[LAPLACIAN] order: {ord}, observed order: {obs}")
    assert np.allclose(obs, ord, rtol=1e-1)

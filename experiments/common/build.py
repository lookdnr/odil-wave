from typing import Tuple, Union

from .config import RunConfig

from odil_wave import (
    Grid,
    SheppLoganModel,
    HomogeneousModel,
    OverDensityModel,
    Wavefield,
    Sources,
    Receivers,
    WaveEquation,
    ForwardLoss,
    Problem,
    GaussNewtonOptimiser,
    LBFGSB,
)

from odil_wave.models.base import VelocityModel
from odil_wave.models import CustomModel

import numpy as np
import h5py

MODELS = {
    "shepp-logan": SheppLoganModel,
    "homogeneous": HomogeneousModel,
    "inclusion": OverDensityModel,
}


def build_source(cfg: RunConfig, grid: Grid) -> Sources:
    return Sources(grid, n_sources=1, source_locs=(cfg.source_loc,), f0=cfg.f0)


def build_receivers(cfg: RunConfig, grid: Grid) -> Receivers:
    return Receivers(
        grid,
        mode=cfg.recv_mode,
        receiver_locs=cfg.recv_locs,
        n_receivers=cfg.n_recvs,
        a_frac=cfg.a_frac,
        b_frac=cfg.b_frac,
        ring_centre=cfg.ring_centre,
    )


def build_grid(cfg: RunConfig) -> Grid:
    return Grid(
        xmin=cfg.xmin,
        xmax=cfg.xmax,
        ymin=cfg.ymin,
        ymax=cfg.ymax,
        nx=cfg.nx,
        ny=cfg.ny,
        t_max=cfg.t_max,
        c_min=cfg.c_min,
        c_max=cfg.c_max,
        cfl_safety=cfg.cfl_safety,
        allow_unstable=cfg.allow_unstable,
    )


def load_h5(path="../../alpha2D-TrueModel.h5") -> np.ndarray:
    """Load model from .h5 file. Defaults to proprietary Sonalis brain atlas"""
    with h5py.File(path, "r") as f:
        return f["data"][()]  # type: ignore


def build_model(cfg: RunConfig, grid: Grid) -> VelocityModel:
    if cfg.model == "custom":
        c = load_h5()
        model = CustomModel(grid, c)

        # must check velocities match
        assert cfg.c_min == model.c_min
        assert cfg.c_max == model.c_max
        return model
    return MODELS[cfg.model](grid, **cfg.model_kwargs)


def built_opt(cfg: RunConfig, loss: ForwardLoss) -> Union[GaussNewtonOptimiser, LBFGSB]:
    meth = cfg.method
    return (
        GaussNewtonOptimiser(loss, outer_maxiter=cfg.maxiter)
        if meth == "paradiag" or meth == "gmres"
        else LBFGSB(loss)
    )


def build_problem(
    cfg: RunConfig,
) -> Tuple[
    Grid,
    VelocityModel,
    Sources,
    Receivers,
    Wavefield,
    WaveEquation,
    Union[GaussNewtonOptimiser, LBFGSB],
]:
    """Build the components of an experiment run, the wavefield and optimiser"""
    grid = build_grid(cfg)
    model = build_model(cfg, grid)
    src = build_source(cfg, grid)
    recvs = build_receivers(cfg, grid)
    wf = Wavefield(grid)
    we = WaveEquation(wf, model, time_order=cfg.time_order, space_order=cfg.space_order)
    loss = ForwardLoss(Problem(we, src))
    opt = built_opt(cfg, loss)
    return (
        grid,
        model,
        src,
        recvs,
        wf,
        we,
        opt,
    )

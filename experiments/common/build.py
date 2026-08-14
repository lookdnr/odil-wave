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
    )


def build_model(cfg: RunConfig, grid: Grid) -> VelocityModel:
    return MODELS[cfg.model](grid, **cfg.model_kwargs)


def built_opt(cfg: RunConfig, loss: ForwardLoss) -> Union[GaussNewtonOptimiser, LBFGSB]:
    meth = cfg.method
    return (
        GaussNewtonOptimiser(loss)
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

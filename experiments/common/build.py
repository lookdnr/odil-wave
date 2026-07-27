from typing import Tuple

from .config import RunConfig

from odil_wave import (
    Grid,
    SheppLoganModel,
    HomogeneousModel,
    OverDensityModel,
    Wavefield,
    Sources,
    WaveEquation,
    ForwardLoss,
    Problem,
    GaussNewtonOptimiser,
)

from odil_wave.models.base import VelocityModel

MODELS = {
    "shepp-logan": SheppLoganModel,
    "homogeneous": HomogeneousModel,
    "inclusion": OverDensityModel,
}


def build_problem(
    cfg: RunConfig,
) -> Tuple[Grid, VelocityModel, Sources, Wavefield, GaussNewtonOptimiser]:
    """Build the components of an experiment run, the wavefield and optimiser"""
    grid = build_grid(cfg)
    model = build_model(cfg, grid)
    src = Sources(grid, n_sources=1, source_locs=(cfg.source_loc,), f0=cfg.f0)
    wf = Wavefield(grid)
    we = WaveEquation(wf, model, time_order=cfg.time_order, space_order=cfg.space_order)
    return grid, model, src, wf, GaussNewtonOptimiser(ForwardLoss(Problem(we, src)))


def build_grid(cfg: RunConfig) -> Grid:
    return Grid(
        xmin=cfg.xmin,
        xmax=cfg.xmax,
        ymin=cfg.ymin,
        ymax=cfg.ymax,
        nx=cfg.nx,
        ny=cfg.ny,
        c_min=cfg.c_min,
        c_max=cfg.c_max,
        cfl_safety=cfg.cfl_safety,
    )


def build_model(cfg: RunConfig, grid: Grid) -> VelocityModel:
    return MODELS[cfg.model](grid, **cfg.model_kwargs)

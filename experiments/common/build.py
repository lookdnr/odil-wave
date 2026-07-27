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

MODELS = {
    "shepp-logan": SheppLoganModel,
    "homogeneous": HomogeneousModel,
    "over-density": OverDensityModel,
}


def build(cfg: RunConfig) -> Tuple[Wavefield, GaussNewtonOptimiser]:
    """Build the components of an experiment run, the wavefield and optimiser"""
    grid = Grid(
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

    model = MODELS[cfg.model](grid, **cfg.model_kwargs)
    src = Sources(grid, n_sources=1, source_locs=(cfg.source_loc,), f0=cfg.f0)
    wf = Wavefield(grid)
    we = WaveEquation(wf, model, time_order=cfg.time_order, space_order=cfg.space_order)
    return wf, GaussNewtonOptimiser(ForwardLoss(Problem(we, src)))

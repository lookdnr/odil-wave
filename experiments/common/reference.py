from examples.seismic import Model, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver

from .config import RunConfig
from .build import build_grid, build_model, build_receivers

from time import perf_counter
import numpy as np


def run_reference(cfg: RunConfig, return_u: bool = False):
    grid = build_grid(cfg)
    velocity = build_model(cfg, grid)
    recvs = build_receivers(cfg, grid)

    # derive nbl from required ppw
    min_wavelength = velocity.c_min / cfg.f0
    nbl = int(2 * min_wavelength / grid.dx)

    model = Model(
        vp=velocity.c,
        origin=(cfg.xmin, cfg.ymin),
        spacing=(grid.dx, grid.dy),
        shape=(cfg.nx, cfg.ny),
        space_order=cfg.space_order,
        nbl=nbl,
        bcs="damp",
    )
    geom = AcquisitionGeometry(
        model,
        recvs.recv_xy,
        np.asarray([cfg.source_loc]),
        t0=0.0,
        tn=grid.t_max,
        src_type="Ricker",
        f0=cfg.f0,
    )

    solver = AcousticWaveSolver(model, geom, space_order=cfg.space_order)

    start = perf_counter()
    rec, u, _ = solver.forward(dt=grid.dt)
    end = perf_counter()

    wall_clock = end - start
    return rec.data, wall_clock, (u if return_u else None)

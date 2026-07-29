from examples.seismic import Model, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver

from .config import RunConfig
from .build import build_grid, build_model

from time import perf_counter
import numpy as np


def build_devito(cfg: RunConfig, rec_coords: np.ndarray, nbl: int = 20):
    grid = build_grid(cfg)
    vmodel = build_model(cfg, grid)
    model = Model(
        vp=vmodel.c.astype(np.float32),
        origin=(cfg.xmin, cfg.ymin),
        spacing=(grid.dx, grid.dy),
        shape=(cfg.nx, cfg.ny),
        space_order=cfg.space_order,
        nbl=nbl,
        bcs="damp",
    )
    src_xy = np.asarray([cfg.source_loc])
    # dt=None corresponds to Devito's critical_dt
    geom = AcquisitionGeometry(
        model,
        np.asarray(rec_coords),
        src_xy,
        t0=0.0,
        tn=grid.t_max,
        src_type="Ricker",
        f0=cfg.f0,
    )
    return model, geom, AcousticWaveSolver(model, geom, space_order=cfg.space_order)


def run_reference(cfg, rec_coords, dt=None, nbl=20, return_u=False) -> dict:
    model, geom, solver = build_devito(
        cfg, rec_coords, nbl=nbl
    )  # geom locked to critical_dt
    dt_used = model.critical_dt if dt is None else dt
    start = perf_counter()
    rec, u, _ = solver.forward(dt=dt_used)
    wall = perf_counter() - start
    nt = rec.data.shape[0]
    t = np.arange(nt) * dt_used  # use operator clock, not the geom.time_axis
    return dict(
        traces=rec.data,
        t=t,
        dt=dt_used,
        wall=wall,
        u=(u if return_u else None),
        geom=geom,
    )

from examples.seismic import Model, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver

from .config import RunConfig
from .build import build_grid, build_model

from time import perf_counter
import numpy as np


def build_devito(cfg: RunConfig, rec_coords: np.ndarray, dt: float | None = None, nbl: int = 20):
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
        interpolation="sinc",
        r=4,  # half width for Kaiser-window
    )

    dt_used = dt if dt is not None else model.critical_dt
    geom = geom.resample(dt_used) # resample to actual dt used

    solver = AcousticWaveSolver(model, geom, space_order=cfg.space_order)
    return model, geom, solver, dt_used 


def run_reference(cfg, rec_coords, dt=None, nbl=20, return_u=False) -> dict:
    """Run the reference solver: Devito"""
    model, geom, solver, dt_used = build_devito(
        cfg, rec_coords, dt, nbl=nbl
    )
    
    start = perf_counter()
    rec, u, summary = solver.forward(dt=dt_used)
    wall = perf_counter() - start

    nt = rec.data.shape[0]
    t = np.arange(nt) * dt_used  # use operator clock, not the geom.time_axis

    # use Devito's own profiler
    entries = [v for v in summary.values() if hasattr(v, "time")]
    kernel_time = float(sum(e.time for e in entries))
    gpointss = float(  # g points per second: giga grid point updates / s
        sum((getattr(e, "gpointss", 0) or 0) for e in entries)
    )

    u_out = None
    # crop to interior if return u 
    if return_u:
        interior = np.array(u.data)[:, nbl:nbl + cfg.nx, nbl:nbl + cfg.ny]
        u_out = interior.reshape(interior.shape[0], -1) # reshape to (nt, nx*ny)

    # check solution is finite
    # only check u_out if it exists
    finite = bool(np.all(np.isfinite(rec.data))) and (bool(np.all(np.isfinite(u_out))) if u_out is not None else True)

    return dict(
        traces=rec.data,
        t=t,
        dt=dt_used,
        wall=wall,
        kernel_time=kernel_time,
        gpointss=gpointss,
        u=u_out,
        finite=finite,
        geom=geom,
    )

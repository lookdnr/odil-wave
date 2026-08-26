import argparse
import json
from dataclasses import replace

import numpy as np

from common import (
    RunConfig,
    build_problem,
    run_optimiser,
    run_reference,
    analytical_traces,
)
from common.compare import pre_reflection_mask
from odil_wave.metrics import normalised_trace_rel_l2
from wave_specific.dsp import field_growth
from odil_wave.grid.utils import nodes_for_ppw

XMAX_HOMOG = 0.1
XMAX_SL = 0.35
F0 = 100e3
C_MIN = 1500.0
PPW = 13.0
N_HOMOG = nodes_for_ppw(XMAX_HOMOG, F0, PPW, C_MIN)
N_SL = nodes_for_ppw(XMAX_SL, F0, PPW, C_MIN)

BASE = RunConfig(
    nx=N_HOMOG,
    ny=N_HOMOG,
    xmin=0.0,
    xmax=XMAX_HOMOG,
    ymin=0.0,
    ymax=XMAX_HOMOG,
    c_min=C_MIN,
    c_max=C_MIN,
    cfl_safety=0.7,  # overwritten per run
    time_order=2,
    space_order=2,  # 2nd order -> cfl_safety=1.0 is max
    f0=F0,
    source_loc=(XMAX_HOMOG / 2, XMAX_HOMOG / 2),
    recv_mode="custom",
    recv_locs=((XMAX_HOMOG * 0.8, XMAX_HOMOG / 2),),
    n_recvs=1,
    model="homogeneous",
    method="paradiag",
    alpha=1e-3,
    rtol=1e-8,
    allow_unstable=True,
)

SL_BASE = replace(
    BASE,
    nx=N_SL,
    ny=N_SL,
    xmax=XMAX_SL,
    ymax=XMAX_SL,
    model="shepp-logan",
    c_min=1500.0,  # background_c
    c_max=2800.0,
    contrast=1300.0,  # 1500 + 1300*1.0 = 2800 = c_max above
    interior_fill=0.65,
    mask_skull=False,
    phantom_centre_frac=(0.95, 0.1),
    source_loc=(0.165, 0.20),
    recv_locs=((0.29, 0.05),),
)

# 17.2 is at approx nyquist for SL config
FIELD_SAVE_CFLS = (0.7, 1.0, 1.3, 3.0, 5.0, 10.0, 17.2)


def _should_save_field(cfl_safety: float) -> bool:
    return any(abs(cfl_safety - v) < 1e-6 for v in FIELD_SAVE_CFLS)


def safe_run_devito(cfg, recv_xy, save, dt):
    """Catch exceptions and flag non-finite output rather than letting one
    unstable config kill the whole sweep"""
    try:
        ref = run_reference(cfg, recv_xy, save=save, r=2, dt=dt)
    except Exception as e:
        return dict(
            traces=None,
            t=None,
            field=None,
            wall=float("nan"),
            finite=False,
            error=str(e),
        )

    ref["finite"] = bool(np.all(np.isfinite(ref["traces"])))
    ref["error"] = None
    return ref


def run(cfl_safety: float, save: bool, which: str = "homog"):
    model = {"homog": BASE, "sl": SL_BASE}[which]
    cfg = replace(model, cfl_safety=cfl_safety)
    grid, _, src, recvs, _, _, opt = build_problem(cfg)

    # figure out whether or not to save
    save_field = cfg.model != "homogeneous" and _should_save_field(cfl_safety)

    # ODIL
    res = run_optimiser(cfg, opt, n_workers=32)
    U_odil = res.res.solution.U
    growth_odil = field_growth(U_odil, grid.dt)

    if cfg.model == "homogeneous":
        d_odil = recvs.extract_observations(U_odil)
        ana_odil = analytical_traces(src, recvs, grid.t, cfg.c_min)
        mask_odil = pre_reflection_mask(src, recvs, grid.t, cfg.c_min)
        err_odil = float(
            normalised_trace_rel_l2(d_odil[mask_odil], ana_odil[mask_odil])
        )
    else:
        err_odil = float("nan")

    # Devito
    ref = safe_run_devito(cfg, recvs.recv_xy, save_field, grid.dt)
    if ref["traces"] is not None:
        d_dev, t_dev = ref["traces"], ref["t"]
        growth_dev = field_growth(d_dev, grid.dt)  # type: ignore

        if cfg.model == "homogeneous":
            ana_dev = analytical_traces(src, recvs, t_dev, cfg.c_min)  # type: ignore
            mask_dev = pre_reflection_mask(src, recvs, t_dev, cfg.c_min)  # type: ignore
            err_dev = float(
                normalised_trace_rel_l2(
                    d_dev[mask_dev], ana_dev[mask_dev]  # type: ignore
                )
            )
        else:
            err_dev = float("nan")
    else:
        growth_dev = dict(
            max_u=[], growth_rate=float("nan"), r2=float("nan"), finite=False
        )
        err_dev = float("nan")

    # save field
    if save_field:
        cfl_str = f"{cfl_safety:.2f}".replace(".", "p")
        field_path = a.out.rsplit(".", 1)[0] + f"_field_{cfl_str}"
        res.res.save(field_path + "_odil", with_field=True)

        if ref["field"] is not None:
            np.savez_compressed(field_path + "_devito.npz", u=ref["field"])

    row = dict(
        cfl_safety=cfl_safety,
        which=which,
        nx=cfg.nx,
        nt=grid.nt,
        wall=res.wall,
        iters=res.iters,
        converged=res.converged,
        finite_odil=growth_odil["finite"],
        growth_rate_odil=growth_odil["growth_rate"],
        r2_odil=growth_odil["r2"],
        err_odil=err_odil,
        finite_dev=growth_dev["finite"],
        growth_rate_dev=growth_dev["growth_rate"],
        r2_dev=growth_dev["r2"],
        err_dev=err_dev,
        devito_error=ref["error"],
    )
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfl-safety", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--save", action="store_true", required=False)
    p.add_argument("--which", choices=["homog", "sl"], default="homog")
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()

    if not a.dry:
        row = run(a.cfl_safety, a.save, a.which)
        with open(a.out, "a") as f:
            f.write(json.dumps(row, default=float) + "\n")

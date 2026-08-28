import argparse
from dataclasses import replace

import numpy as np

from common import (
    RunConfig,
    build_problem,
    run_reference,
    run_optimiser,
    ricker,
    DispersionResult,
)

from common.compare import pre_reflection_mask
from wave_specific import (
    ray_receiver_locs,
    trace_spectra,
    pw_phase_velocity,
    compute_attenuation,
)
from odil_wave.grid.utils import nodes_for_ppw
from common import save

SOURCE_LOC = (0.1, 0.2)

# receiver set up
RADII = np.arange(0.05, 0.15, 0.005)  # src-rec distances, > wavelength
ANGLES = [0.0, 15.0, 30.0, 45.0]  # deg
LOCS, RAYS = ray_receiver_locs(SOURCE_LOC, ANGLES, RADII)
RECV_LOCS = tuple(loc for loc in LOCS)  # cast to tuple for type hint

PPW_VALUES = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
BASE = RunConfig(
    nx=135,
    ny=135,  # overwritten per ppw below
    xmin=0.0,
    xmax=0.4,
    ymin=0.0,
    ymax=0.4,
    c_min=1500.0,
    c_max=1500.0,  # homogeneous
    cfl_safety=0.7,
    time_order=2,
    space_order=2,
    f0=50e3,
    source_loc=SOURCE_LOC,
    recv_mode="custom",
    recv_locs=RECV_LOCS,
    n_recvs=len(RECV_LOCS),
    model="homogeneous",
    method="paradiag",
    alpha=1e-3,
)


def run(cfg, n_workers):
    """Colelct results for the given config"""
    c, f0 = cfg.c_min, cfg.f0
    grid, _, src, recvs, _, _, opt = build_problem(cfg)

    # run, get observations
    res = run_optimiser(cfg, opt, n_workers=n_workers)
    solve_res = res.res
    d_odil = recvs.extract_observations(solve_res.solution.U)

    # run devito, no need to account for JIT compile since we dc about time
    ref = run_reference(cfg, recvs.recv_xy, dt=grid.dt, r=2)
    d_dev = ref["traces"]

    # compute analytical traces and masks
    mask = pre_reflection_mask(src, recvs, grid.t, c)

    # compute source spectrum
    src_spectrum = np.abs(np.fft.rfft(ricker(grid.t, f0, 1 / f0)))

    # create fiter: exclude bottom 10% of amplitude spectrum
    band = src_spectrum > 0.1 * src_spectrum.max()

    n_radii = len(RADII)
    angles_out = {}

    # for each angle, compute phase velocity and attenutation coeff
    for i, angle in enumerate(ANGLES):
        cols = slice(i * n_radii, (i + 1) * n_radii)

        omegas, ffts_odil = trace_spectra(d_odil[:, cols], grid.dt, mask[:, cols])
        _, ffts_dev = trace_spectra(d_dev[:, cols], grid.dt, mask[:, cols])

        # phase vels
        v_odil, _, _ = pw_phase_velocity(omegas, ffts_odil, RADII)
        v_dev, _, _ = pw_phase_velocity(omegas, ffts_dev, RADII)

        # attenuation coeffs
        alpha_odil = compute_attenuation(omegas, ffts_odil, RADII)
        alpha_dev = compute_attenuation(omegas, ffts_dev, RADII)

        window_us = mask[:, cols].sum(axis=0) * grid.dt * 1e6

        angles_out[angle] = dict(
            radii=RADII.tolist(),
            freqs=omegas[band].tolist(),
            v_odil=v_odil[band].tolist(),
            v_dev=v_dev[band].tolist(),
            alpha_odil=alpha_odil[band].tolist(),
            alpha_dev=alpha_dev[band].tolist(),
            window_us=window_us.tolist(),
        )

    return DispersionResult(
        ppw=cfg.ppw, nx=cfg.nx, nt=grid.nt, dt=grid.dt, angles=angles_out
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--dry", required=False, action="store_true")
    p.add_argument("--workers", required=False, default=128)
    a = p.parse_args()

    results = []
    for ppw in PPW_VALUES:
        n = nodes_for_ppw(BASE.xmax - BASE.xmin, BASE.f0, ppw, BASE.c_min)
        cfg = replace(BASE, nx=n, ny=n)
        if not a.dry:
            results.append(run(cfg, a.workers))
            save(results, a.out)  # checkpoint after every point

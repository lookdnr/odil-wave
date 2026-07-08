from odil_wave import (
    Grid,
    SheppLoganModel,
    Sources,
    Wavefield,
    WaveEquation,
    Problem,
    ForwardLoss,
    GaussNewtonOptimiser,
)
import numpy as np
import time
import argparse

"""
A script for running the forward problem over the Shepp Logan Phantom model.
"""

# 300 x 300 grid
NX = 300
NY = 300

# 25cm x 25cm domain
XMIN = 0.0
XMAX = 0.25
YMIN = 0.0
YMAX = 0.25

# wave speed and cfl
BACKGROUND_C = 1500.0  # water
C_MAX = 1800.0
CFL_SAFETY = 0.9

# source control
N_SOURCES = 1
SOURCE_LOCS = ((0.03, 0.125),)  # in physical coords, just outside skull
F0 = 300000  # Hz

# model
MASK_SKULL = True
INTERIOR_FILL = 1.0

# discretisation
TIME_ORDER = 2
SPACE_ORDER = 8

# optimiser
METHOD = "paradiag"
ALPHA = 1e-3  # alpha constant for circulant preconditioner

# plotting and save
SAVE_GIF = True
GIF_TITLE = "300kHz source over soft Shepp Logan Phantom"
GIF_OUTFILE = "SheppLogan-baseline.gif"

SAVE_DATA = True
DATA_OUTFILE = "SheppLogan-baseline.npy"

# dry run: set true if you don't want to optimise to check setup OK
DRY_RUN = False


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline direct-LU solve.")
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Dry run: skip the expensive solve for testing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DRY_RUN = args.dry

    if DRY_RUN:
        SAVE_DATA = False
        SAVE_GIF = False

    print("Setting up...\n")

    setup_start = time.perf_counter()

    grid = Grid(
        nx=NX,
        ny=NY,
        xmin=XMIN,
        xmax=XMAX,
        ymin=YMIN,
        ymax=YMAX,
        c_min=BACKGROUND_C,
        c_max=C_MAX,
        cfl_safety=CFL_SAFETY,
    )
    print("Grid OK")
    print(grid.summary)
    print()

    model = SheppLoganModel(
        grid,
        background_c=BACKGROUND_C,
        interior_fill=INTERIOR_FILL,
        mask_skull=MASK_SKULL,
    )
    print("Model OK")

    source = Sources(grid, n_sources=N_SOURCES, source_locs=SOURCE_LOCS, f0=F0)
    print("Sources OK")

    wavefield = Wavefield(grid)
    print("Wavefield OK")

    equation = WaveEquation(
        wavefield, model, time_order=TIME_ORDER, space_order=SPACE_ORDER
    )
    print("Equation OK")

    problem = Problem(equation, source)
    print("Problem OK")

    loss = ForwardLoss(problem)
    print("Loss OK")

    optimiser = GaussNewtonOptimiser(loss)
    print("Optimiser OK\n")

    setup_end = time.perf_counter()

    setup_duration = setup_end - setup_start
    print(f"Setup complete in {setup_duration:.9f} s\n")

    opt_duration = 0.0
    if not DRY_RUN:
        print("Optimising...")
        opt_start = time.perf_counter()
        result = optimiser.minimise(wavefield, method=METHOD, alpha=ALPHA)
        opt_end = time.perf_counter()

        opt_duration = opt_end - opt_start
        print("Finished optimising.")
        print("Converged:", result.success)
        print(f"Optimisation converged in {opt_duration:.9f} s\n")

    if SAVE_DATA:
        print("Saving data...")
        np.save(DATA_OUTFILE, result.solution.U)
        print("Done\n")

    if SAVE_GIF:
        print("Animating...")
        result.solution.animate(title=GIF_TITLE)
        print("GIF saved to", GIF_OUTFILE)

    print("Run complete.")
    print(f"Total duration: {setup_duration + opt_duration:.9f} s")

    if SAVE_DATA:
        print("Data saved to", DATA_OUTFILE)

    if SAVE_GIF:
        print("GIF saved to", GIF_OUTFILE)

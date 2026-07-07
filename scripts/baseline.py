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

"""
A script for running the forward problem over the Shepp Logan Phantom model.
"""

# 300 x 300 grid
nx = 300
ny = 300

# 25cm x 25cm domain
xmin = 0.0
xmax = 0.25
ymin = 0.0
ymax = 0.25

# wave speed and cfl
background_c = 1500.0  # water
c_ref = 1800.0
cfl_safety = 0.9

# source control
n_sources = 1
source_loc = ((0.03, 0.125),)  # in physical coords, just outside skull
f0 = 300000  # Hz

# model
mask_skull = True
interior_fill = 1.0

# discretisation
time_order = 2
space_order = 8

# optimiser
method = "paradiag"
alpha = 1e-3  # alpha constant for circulant preconditioner

# plotting and save
gif_title = "300kHz source over soft Shepp Logan Phantom"
gif_outfile = "SheppLogan-baseline.gif"

save = True
data_outfile = "SheppLogan-baseline.npy"

print("Setting up...\n")

setup_start = time.perf_counter()

grid = Grid(
    nx=nx,
    ny=ny,
    xmin=xmin,
    xmax=xmax,
    ymin=ymin,
    ymax=ymax,
    c_ref=c_ref,
    cfl_safety=cfl_safety,
)
print("Grid OK")
print(grid.summary)
print()

model = SheppLoganModel(
    grid, background_c=background_c, interior_fill=interior_fill, mask_skull=mask_skull
)
print("Model OK")

source = Sources(grid, n_sources=n_sources, source_locs=source_loc, f0=f0)
print("Sources OK")

wavefield = Wavefield(grid)
print("Wavefield OK")

equation = WaveEquation(
    wavefield, model, time_order=time_order, space_order=space_order
)
print("Equation OK")

problem = Problem(equation, source)
print("Problem OK")

loss = ForwardLoss(problem)
print("Loss OK")

optimiser = GaussNewtonOptimiser(loss)
print("Optimiser OK\n")

setup_end = time.perf_counter()

print(f"Setup complete in {setup_end - setup_start:.9f} s")

print("Optimising...")

opt_start = time.perf_counter()
result = optimiser.minimise(wavefield, method=method, alpha=alpha)
opt_end = time.perf_counter()
print("Finished optimising.")
print("Converged:", result.success)
print(f"Optimisation converged in {opt_end - opt_start:.9f} s\n")

if save:
    print("Saving data...")
    np.save(data_outfile, result.solution.U)
    print("Done\n")

print("Animating...")
result.solution.animate(title=gif_title)
print("GIF saved to", gif_outfile)

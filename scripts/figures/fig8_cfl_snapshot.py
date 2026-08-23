import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import skimage.measure as skm
from dataclasses import replace
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

from odil_wave import Wavefield

from common.build import build_grid, build_model # type: ignore

from exp4_cfl_sweep import SL_BASE # type: ignore

PREFIX = "results/wave/cfl_sweep_sl_field"
TIME_LEVELS = [15, 50, 80, 110]
CFL_LOW, CFL_HIGH = 0.7, 1.3

FIGURE = "fig8_cfl_snapshots.png"

def load_field_wavefield(cfl_safety, solver):
    """Load a saved CFL-sweep field snapshot into a Wavefield."""
    cfl_str = f"{cfl_safety:.2f}".replace(".", "p")
    cfg = replace(SL_BASE, cfl_safety=cfl_safety)
    grid = build_grid(cfg)

    data = np.load(f"{PREFIX}_{cfl_str}_{solver}.npz")["u"]

    return Wavefield(grid, init_amplitude=data)

def build_fine_model(base_cfg, fine_n=500):
    """High resolution model for plotting contours"""
    fine_cfg = replace(base_cfg, nx=fine_n, ny=fine_n)
    fine_grid = build_grid(fine_cfg)
    return build_model(fine_cfg, fine_grid)

def add_model_contours(ax, model, colour="k", alpha=0.5, linewidth=0.5, linestyle="-"):
    """Plot contours of the model. Forms a mask for levels in the model,
    labels them, then extracts contours using marchiong squares from skimage"""
    c = model.c
    (xmin, xmax), (ymin, ymax) = model.grid.extent
    nx, ny = c.shape

    c_int = c.astype(int) # cast to int to avoid granular contours

    # extract velocity levels against background
    background = c_int.min()
    levels = np.unique(c_int.astype(int))
    levels = levels[levels != background] # filter for non-bg levels

    # for each level, create mask, label regions, adn build contour
    for lvl in levels:
        mask = np.isclose(c_int, lvl)
        region_labels = np.array(skm.label(mask))

        # build mask for region in this level
        for region_id in range(1, region_labels.max() + 1):
            region_mask = region_labels == region_id

        # build contour and plot
        for contour in skm.find_contours(region_mask.astype(float)):
            xs = xmin + contour[:, 0] / (nx - 1) * (xmax - xmin)
            ys = ymin + contour[:, 1] / (ny - 1) * (ymax - ymin)
            ax.plot(xs, ys, color=colour, linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=3)


def plot_cfl_snapshot(gap_frac=0.1, label_frac=0.1):
    """Plot panel of ODIL vs Devito snapshots at specific time levels and CFL numbers"""

    # set up panel
    ncols = len(TIME_LEVELS)

    # 4 rows, 2 each for devito and odil at specified cfls
    row_specs = [
        ("label", f"CFL = {CFL_LOW}"),
        ("ODIL", CFL_LOW), ("Devito", CFL_LOW),
        None, # visual gap
        ("label", f"CFL = {CFL_HIGH}"),
        ("ODIL", CFL_HIGH), ("Devito", CFL_HIGH),
    ]

    height_ratios = [
        gap_frac if spec is None else (label_frac if spec[0] == "label" else 1.0)
        for spec in row_specs
    ]

    nrows = len(row_specs)

    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(nrows, ncols, figure=fig, height_ratios=height_ratios, hspace=0.1, wspace=0.01)

    # build fine model to show SL phantom contour
    model = build_fine_model(SL_BASE)

    for i, spec in enumerate(row_specs):
        if spec is None: # skip centre
            continue

        # top level labels above rows
        if spec[0] == "label":
            ax = fig.add_subplot(gs[i, :])
            ax.axis("off")
            ax.text(0.5, 0.5, spec[1], transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=12, fontweight="bold")
            continue

        # load wavefield from spec
        solver_label, cfl = spec
        solver = "odil" if solver_label == "ODIL" else "devito"
        wf = load_field_wavefield(cfl, solver)

        # reshape for plotting
        amp = wf.U.reshape(wf.grid.nt, *wf.grid.shape)
        (xmin, xmax), (ymin, ymax) = wf.grid.extent
        t = wf.grid.t

        # plot at time intervals, adding a subplot for each
        for j, t_us in enumerate(TIME_LEVELS):
            idx = int(np.argmin(np.abs(t - t_us * 1e-6)))
            frame = amp[idx]
            peak = np.abs(frame).max()
            ax = fig.add_subplot(gs[i, j])

            # plot and add SL phantom contour
            ax.imshow(frame.T, origin="lower", extent=(xmin, xmax, ymin, ymax), cmap="RdBu_r", vmin=-peak, vmax=peak)
            add_model_contours(ax, model)

            ax.set_xticks([]); ax.set_yticks([])

            if i in (1, 5):
                ax.set_title(rf"{t_us:.0f}$\mu$s", fontsize=9)

            if j == 0:
                ax.set_ylabel(solver_label, fontsize=9, rotation=0, ha="right", va="center")


    return fig

if __name__ == "__main__":
    fig = plot_cfl_snapshot()
    fig.savefig(FIGURE, dpi=200)
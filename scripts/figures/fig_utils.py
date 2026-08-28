import json
from pathlib import Path
import pandas as pd
import pickle
import numpy as np
import skimage.measure as skm
from dataclasses import replace
import sys

from odil_wave import Wavefield

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

from common.build import build_grid, build_model  # type: ignore
import copy


def load_jsonl(which: Path, path: Path) -> pd.DataFrame:
    """Load reuslts jsonl from path to df"""
    with open(path / which) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows)


def bytes_to_gib(result: pd.DataFrame):
    """Convert memory cols from bytes to gigibytes"""
    for col in [
        "peak_rss",
        "self_rss",
        "child_rss",
        "cached_total_bytes",
        "uncached_peak_bytes",
        "per_mode_bytes_mean",
        "per_mode_bytes_max",
    ]:
        if col in result.columns:
            result[col + "_GiB"] = result[col] / 2**30
    return result


def load_pkl(path):
    """Load an accuracy sweep from pickled file"""

    with open(path, "rb") as f:
        return pickle.load(f)


def load_field_wavefield(cfg, prefix, cfl_safety, solver):
    """Load a saved CFL-sweep field snapshot into a Wavefield."""
    cfl_str = f"{cfl_safety:.2f}".replace(".", "p")
    cfg = replace(cfg, cfl_safety=cfl_safety)
    grid = build_grid(cfg)

    data = np.load(f"{prefix}_{cfl_str}_{solver}.npz")["u"]

    # saved field's nt may not match a rebuilt Grid's nt if
    # SL_BASE has changed since the file was written
    n_spatial = grid.nx * grid.ny
    nt_actual, remainder = divmod(data.size, n_spatial)
    if remainder != 0:
        raise ValueError(
            f"saved field size {data.size} isn't a multiple of nx*ny={n_spatial}"
        )

    if nt_actual != grid.nt:
        grid = copy.copy(grid)
        grid.nt = nt_actual
        grid.t = np.linspace(0.0, grid.dt * (nt_actual - 1), nt_actual)

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

    c_int = c.astype(int)  # cast to int to avoid granular contours

    # extract velocity levels against background
    background = c_int.min()
    levels = np.unique(c_int.astype(int))
    levels = levels[levels != background]  # filter for non-bg levels

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
            ax.plot(
                xs,
                ys,
                color=colour,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                zorder=3,
            )


def reshape(wf):
    """Reshape flattened wavefield to plottable image"""
    amp = wf.U.reshape(wf.grid.nt, *wf.grid.shape)
    return amp, wf.grid.extent, wf.grid.t

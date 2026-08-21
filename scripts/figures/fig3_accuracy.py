import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
import sys

# make experiments/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

from common import load  # type: ignore

# set dirs
ROOT = Path(__file__).resolve().parents[2]  # ../../
RESULTS = ROOT / "results" / "accuracy" / "sweep_50khz.pkl"
FIGURE = ROOT / "fig3_accuracy.png"

COL = {"odil": "royalblue", "dev": "darkorange"}


def to_frame(results):
    """Gather results into a df"""
    return (
        pd.DataFrame(r.metrics for r in results)
        .sort_values("dof")
        .reset_index(drop=True)
    )


def plot_error_vs_dof(df, ax):
    ax.loglog(df.dof, df.err_odil, "o-", color=COL["odil"])
    ax.loglog(df.dof, df.err_dev, "s--", color=COL["dev"])
    ax.set(xlabel=r"DOF $(N_x \times N_y \times N_t)$", ylabel="Global Error Norm")
    ax.grid(True, which="both", alpha=0.3)

    ax.set_yticks([0.01, 0.02, 0.03, 0.04, 0.05])
    ax.set_yticklabels(["0.01", "0.02", "0.03", "0.04", "0.05"])


def plot_diff_map(result, ax, highlight_k, cmap="berlin"):
    """Spatial map of relative (ODIL - Devito) error, centered at 0"""
    xy = np.array([rc.xy for rc in result.receivers])
    eo = np.array([rc.err_odil for rc in result.receivers])
    ed = np.array([rc.err_dev for rc in result.receivers])
    rel = (eo - ed) / ed

    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=rel,
        cmap=cmap,
        vmin=-0.5,
        vmax=0.5,
        s=120,
        edgecolor="k",
        linewidth=0.8,
    )
    ax.scatter(0.1, 0.1, marker="*", s=180, color="orangered")

    # indiate overlay receiver
    ax.scatter(
        *xy[highlight_k],
        s=120,
        facecolor="none",
        edgecolor="r",
        linewidth=2.2,
        zorder=5,
    )

    ax.annotate(
        "c",
        xy[highlight_k],
        textcoords="offset points",
        xytext=(12, -3),
        fontsize=18,
        fontweight="bold",
        color="r",
        zorder=5,
    )

    ax.set_aspect("equal")
    ax.set(xlim=(0, 0.2), ylim=(0, 0.2), xlabel="x (m)", ylabel="y (m)")
    return sc


def _max_norm(data: np.ndarray):
    return data / np.max(np.abs(data))


def plot_trace_overlay(result, k, ax):
    to, td = result.trace(k, "odil"), result.trace(k, "dev")
    t, window = td["t"] * 1e6, td["mask"]
    tw0, tw1 = t[window][0], t[window][-1]

    ax.axvspan(tw0, tw1, color="limegreen", alpha=0.10, zorder=0)

    ax.plot(
        to["t"] * 1e6,
        _max_norm(to["numerical"]),
        color=COL["odil"],
        linewidth=5,
        solid_capstyle="round",
        label="ODIL",
    )
    ax.plot(
        td["t"] * 1e6,
        _max_norm(td["numerical"]),
        color=COL["dev"],
        linewidth=2.5,
        solid_capstyle="round",
        label="Devito",
    )
    ax.plot(
        td["t"] * 1e6,
        _max_norm(td["analytical"]),
        "k:",
        linewidth=2,
        label="Analytical",
    )

    ax.text(
        tw1 + 10,
        0.9,
        r"Reflections",
        ha="center",
        va="top",
        fontsize=18,
        clip_on=False,
    )

    ax.annotate(
        "",
        xy=(115, 0.3),
        xytext=(tw1 + 10, 0.7),
        fontsize=18,
        arrowprops=dict(arrowstyle="->", lw=1.5),
        annotation_clip=False,
    )

    ax.text(
        tw1 - 1,
        -0.58,
        ">",
        ha="center",
        va="top",
        fontsize=18,
        clip_on=False,
        color="k",
        alpha=0.8,
    )

    ax.text(
        tw1 - 35,
        -0.58,
        "Pre-reflection window",
        ha="center",
        va="top",
        fontsize=18,
        clip_on=False,
        color="k",
        alpha=0.8,
    )

    ax.text(
        tw0 + 1,
        -0.58,
        "<",
        ha="center",
        va="top",
        fontsize=18,
        clip_on=False,
        color="k",
        alpha=0.8,
    )


def make_accuracy_figure(
    results, map_idx=0, overlay_idx=0, overlay_k=0, cmap="cividis"
):
    """Make the accuracy figure panel from above functions"""
    df = to_frame(results)
    r_map = results[map_idx]  # receiver map config
    r_ovl = results[overlay_idx]  # trace overlay config

    # setup figure and panel
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = gs.GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.0])

    # a) error vs dof
    top = grid[0].subgridspec(1, 3, width_ratios=[1, 2, 1])
    ax_dof = fig.add_subplot(top[0, 1])
    plot_error_vs_dof(df, ax_dof)

    # b) receiver map differences
    # bottom two panels
    bot = grid[1].subgridspec(1, 4, width_ratios=[1, 1, 1, 1])
    ax_diff = fig.add_subplot(bot[0, 0])
    sc = plot_diff_map(r_map, ax_diff, overlay_k)

    fig.colorbar(
        sc,
        ax=ax_diff,
        shrink=0.95,
        pad=0.04,
        label=r"$\mathcal{E}_\mathrm{ODIL} - \mathcal{E}_\mathrm{Dev}$",
        location="top",
    )

    # c) trace
    ax_trace = fig.add_subplot(bot[0, 1:])
    plot_trace_overlay(r_ovl, overlay_k, ax_trace)

    # panel labels
    for ax, lab in zip([ax_dof, ax_diff, ax_trace], "abc"):
        ax.set_title(f"({lab})", loc="left", fontweight="bold", fontsize=14)

    legend_handles = [
        Patch(facecolor=COL["odil"], label="ODIL"),
        Patch(facecolor=COL["dev"], label="Devito"),
        Line2D([0], [0], color="k", linestyle=":", linewidth=4.1, label="Analytical"),
        Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            markersize=20,
            color="orangered",
            markeredgecolor="r",
            label="Source",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=15,
            markeredgewidth=1.0,
            color="0.6",
            markeredgecolor="k",
            fillstyle="none",
            label="Receiver",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
        fontsize=16,
        ncol=5,
    )
    return fig


def main():
    results = load(str(RESULTS))
    fig = make_accuracy_figure(results)

    fig.savefig(FIGURE, dpi=200, bbox_inches="tight", pad_inches=0.1)
    print("Results saved to", str(FIGURE))


if __name__ == "__main__":
    main()

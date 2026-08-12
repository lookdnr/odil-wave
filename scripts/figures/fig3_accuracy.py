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

from accuracy.storage import load  # type: ignore

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


def plot_error_vs_dof(df: pd.DataFrame, ax):
    """Plot error of both methods on log-log space vs analytical soln"""
    ax.loglog(df.dof, df.err_odil, "o-", color=COL["dev"])
    ax.loglog(df.dof, df.err_dev, "s--", color=COL["odil"])

    ax.set(
        xlabel=r"DOF $(N_x \ \times N_y \ \times \ N_t)$", ylabel="Global Error Norm"
    )
    ax.grid(True, which="both", alpha=0.3)


def plot_receiver_map(
    result, ax_odil, ax_dev, cmap="cividis", highlight_ks=(), labels=()
):
    """Plot receiver locations coloured by error"""
    # extract coords and error from AccuracyResult
    xy = np.array([rc.xy for rc in result.receivers])
    eo = np.array([rc.err_odil for rc in result.receivers])
    ed = np.array([rc.err_dev for rc in result.receivers])

    # get limits for shared colorbar
    vmin, vmax = min(eo.min(), ed.min()), max(eo.max(), eo.max())

    for ax, e, name in [(ax_odil, eo, "ODIL"), (ax_dev, ed, "Devito")]:
        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=e,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=120,
            edgecolor="k",
            linewidth=0.8,
        )
        ax.scatter(0.1, 0.1, marker="*", s=180, color="orangered")  # source

        for k, label in zip(highlight_ks, labels):
            ax.scatter(
                *xy[k], s=120, facecolor="none", edgecolor="r", linewidth=1.2, zorder=4
            )

            if label == "d":
                ax.annotate(
                    label,
                    xy[k],
                    textcoords="offset points",
                    xytext=(12, -3),
                    fontsize=18,
                    fontweight="bold",
                    zorder=5,
                )
            else:
                ax.annotate(
                    label,
                    xy[k],
                    textcoords="offset points",
                    xytext=(-25, 0),
                    fontsize=18,
                    fontweight="bold",
                    zorder=5,
                )

        ax.set_aspect("equal")
        ax.set_title(name)

    return sc


def _max_norm(data: np.ndarray):
    return data / np.max(np.abs(data))


def plot_trace_overlays(result, ks, axes):
    """Plot an overlay of trace at receiver indexes ks"""

    for k, ax in zip(ks, axes):
        # extract recorded traces
        to, td = result.trace(k, "odil"), result.trace(k, "dev")

        # plot pre-reflection window
        t, window = td["t"] * 1e6, td["mask"]
        tw0, tw1 = t[window][0], t[window][-1]
        ax.axvspan(tw0, tw1, color="limegreen", alpha=0.10, zorder=0)

        # plot numerical solns
        ax.plot(
            to["t"] * 1e6,
            _max_norm(to["numerical"]),
            label="ODIL",
            color=COL["odil"],
            linewidth=5,
            solid_capstyle="round",
        )
        ax.plot(
            td["t"] * 1e6,
            _max_norm(td["numerical"]),
            label="Devito",
            color=COL["dev"],
            linewidth=2.5,
            solid_capstyle="round",
        )

        # plot analytical soln
        ax.plot(
            td["t"] * 1e6,
            _max_norm(td["analytical"]),
            "k:",
            label="Analytical",
            linewidth=2,
            solid_capstyle="round",
        )

        ax.set_title(f"Receiver {k+1}")

    axes[0].text(
        tw1 - 1,
        1.0,
        r"Reflections",
        ha="center",
        va="top",
        fontsize=18,
        clip_on=False,
    )

    axes[0].annotate(
        "",
        xy=(115, 0.25),
        xytext=(tw1, 0.85),
        fontsize=18,
        arrowprops=dict(arrowstyle="->", lw=1.5),
        annotation_clip=False,
    )

    axes[1].text(
        tw1 - 28,
        1.0,
        "Pre-reflection window",
        ha="center",
        va="top",
        fontsize=18,
        clip_on=False,
        color="limegreen",
        alpha=0.9,
    )


def make_accuracy_figure(
    results, map_idx=0, overlay_idx=0, overlay_ks=(0, 9), cmap="cividis"
):
    """Make the accuracy figure panel from above functions"""
    df = to_frame(results)
    r_map = results[map_idx]  # receiver map config
    r_ovl = results[overlay_idx]  # trace overlay config

    # setup figure and panel
    fig = plt.figure(figsize=(18, 9), constrained_layout=True)
    grid = gs.GridSpec(2, 4, figure=fig, height_ratios=[1.0, 0.95])

    # a) error vs dof
    ax_dof = fig.add_subplot(grid[0, 0:2])
    plot_error_vs_dof(df, ax_dof)

    # b, c) error maps with 1 color bar
    ax_o = fig.add_subplot(grid[0, 2])
    ax_d = fig.add_subplot(grid[0, 3])
    ov_labels = ("d", "e")
    sc = plot_receiver_map(r_map, ax_o, ax_d, cmap, overlay_ks, ov_labels)

    for ax in (ax_o, ax_d):
        ax.set_xlim(0, 0.2)
        ax.set_ylim(0, 0.2)
        ax.set_xlabel("x (m)")

    ax_o.set_ylabel("y (m)")
    fig.colorbar(sc, ax=[ax_o, ax_d], shrink=0.95, label="Trace Error Norm")

    # d,e) trace overlays
    # bottom two panels
    ax_tr = [fig.add_subplot(grid[1, 0:2]), fig.add_subplot(grid[1, 2:4])]
    plot_trace_overlays(r_ovl, overlay_ks, ax_tr)

    for ax in ax_tr:
        ax.set_xlabel(r"t ($\mu$s)")

    ax_tr[0].set_ylabel("Normalised amplitude")

    # panel labels
    for ax, lab in zip([ax_dof, ax_o, ax_d, *ax_tr], "abcde"):
        ax.text(
            -0.12,
            1.2,
            f"({lab})",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
        )

    legend_handles = [
        Patch(facecolor=COL["odil"], label="ODIL"),
        Patch(facecolor=COL["dev"], label="Devito"),
        Line2D([0], [0], color="k", linestyle=":", linewidth=4, label="Analytical"),
        Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            markersize=20,
            color="orangered",
            markeredgecolor="k",
            label="Source",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=15,
            color="0.6",
            markeredgecolor="k",
            label="Receivers",
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

    fig.savefig(FIGURE, dpi=1000, bbox_inches="tight", pad_inches=0.1)
    print("Results saved to", str(FIGURE))


if __name__ == "__main__":
    main()

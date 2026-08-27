from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import json

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "wave"
FIGURE = "fig8_cfl_stability.png"

TITLE_FS = 16
LABEL_FS = 14

COL = {"odil": "royalblue", "devito": "darkorange"}


def load_jsonl(path):
    """Load a .jsonl results file into a list of dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_first_non_finite(cfls, non_finite_mask):
    """Compute position of first non finite element in mask"""
    sorted_positions = np.argsort(cfls)
    sorted_non_finite = non_finite_mask[sorted_positions]
    return sorted_positions[sorted_non_finite][0]


def plot_error_vs_cfl(rows, ax):
    """Relative L2 trace error vs cfl_safety"""
    # extract cfl and error
    cfls = np.array([r["cfl_safety"] for r in rows])
    e_odil = np.array([r["err_odil"] for r in rows], dtype=float)
    e_dev = np.array([r["err_dev"] for r in rows], dtype=float)

    # mask where non finite
    fin_dev = np.array([r["finite_dev"] for r in rows])
    non_finite_dev = fin_dev == False

    # plot errors
    ax.semilogx(cfls, e_odil, "o-", color=COL["odil"])
    ax.semilogx(
        cfls[non_finite_dev != True],
        e_dev[non_finite_dev != True],
        "s-",
        color=COL["devito"],
    )

    # mark non finite wiht crosses
    if non_finite_dev.any():

        fnf = get_first_non_finite(cfls, non_finite_dev)
        ax.scatter(
            cfls[fnf - 1], e_dev[fnf - 1], marker="x", s=120, color="red", zorder=6
        )

    ax.axvline(1.0, color="gray", lw=0.8, ls=":")
    ax.set(xlabel="CFL", ylabel="Relative L2 trace error", title="Error vs CFL")

    xticks = [0.7, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    xticks_labels = [str(tick) for tick in xticks]
    ax.set_xticks(xticks, labels=xticks_labels)
    plt.tight_layout(h_pad=2)
    return ax


def plot_growth_vs_cfl(rows, ax):
    """Growth rate vs cfl_safety"""
    # extract cfl and growth rate
    cfls = np.array([r["cfl_safety"] for r in rows])
    g_odil = np.array([r["growth_rate_odil"] for r in rows], dtype=float)
    g_dev = np.array([r["growth_rate_dev"] for r in rows], dtype=float)

    # mask for where non finite
    fin_dev = np.array([r["finite_dev"] for r in rows])
    non_finite_dev = fin_dev == False

    # plot
    ax.semilogx(cfls, g_odil, "o-", color=COL["odil"])
    ax.semilogx(cfls, g_dev, "s-", color=COL["devito"])

    # mark non finite with cross
    if non_finite_dev.any():
        fnf = get_first_non_finite(cfls, non_finite_dev)

        ax.scatter(
            cfls[fnf - 1], g_dev[fnf - 1], marker="x", s=120, color="red", zorder=6
        )

    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.axvline(1.0, color="gray", lw=0.8, ls=":")
    ax.set(xlabel="CFL", ylabel="Growth rate", title="Field growth vs CFL")
    xticks = [0.7, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    xticks_labels = [str(tick) for tick in xticks]
    ax.set_xticks(xticks, labels=xticks_labels)
    return ax


def plot_r2_vs_cfl(rows, ax):
    """R2 of the growth rate vs cfl_safety"""
    cfls = np.array([r["cfl_safety"] for r in rows])
    r2_odil = np.array([r["r2_odil"] for r in rows], dtype=float)
    r2_dev = np.array([r["r2_dev"] for r in rows], dtype=float)

    fin_dev = np.array([r["finite_dev"] for r in rows])
    non_finite_dev = fin_dev == False

    ax.semilogx(cfls, r2_odil, "o-", color=COL["odil"])
    ax.semilogx(
        cfls[non_finite_dev != True],
        r2_dev[non_finite_dev != True],
        "s-",
        color=COL["devito"],
    )

    if non_finite_dev.any():
        fnf = get_first_non_finite(cfls, non_finite_dev)
        ax.scatter(
            cfls[fnf - 1], r2_dev[fnf - 1], marker="x", s=120, color="red", zorder=6
        )

    ax.axvline(1.0, color="gray", lw=0.8, ls=":")
    ax.set(xlabel="CFL", ylabel="$R^2$ (growth fit)", title="Growth rate fit quality")
    ax.set_ylim(top=1.02)

    xticks = [0.7, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    xticks_labels = [str(tick) for tick in xticks]
    ax.set_xticks(xticks, labels=xticks_labels)
    return ax


def plot_cfl_stability(rows, figsize=(12, 4)):
    """Error and growth rate vs CFL, side by side"""
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    plot_error_vs_cfl(rows, ax=axs[0])
    plot_growth_vs_cfl(rows, ax=axs[1])
    plot_r2_vs_cfl(rows, ax=axs[2])

    legend_handles = [
        Line2D([0], [0], color=COL["odil"], marker="o", label="ODIL"),
        Line2D([0], [0], color=COL["devito"], marker="s", label="Devito"),
        Line2D([0], [0], color="gray", linestyle=":", label="CFL limit"),
        Line2D(
            [0],
            [0],
            color="red",
            marker="x",
            linestyle="None",
            markersize=10,
            label="First non-finite",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        frameon=False,
        fontsize=TITLE_FS,
        ncol=4,
    )

    for ax, lab in zip(axs, "ab"):
        ax.text(
            -0.12,
            1.05,
            f"({lab})",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
            fontsize=14,
        )

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    rows = load_jsonl(RESULTS_DIR / "cfl_sweep_homog.jsonl")
    fig = plot_cfl_stability(rows)

    fig.savefig(FIGURE, dpi=200, bbox_inches="tight")

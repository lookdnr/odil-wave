from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as pltgs
import numpy as np
from common import bytes_to_gib, load_jsonl

RESULTS_DIR = Path("results/performance/")
CACHED = Path("f0_sweep_odil_cached.jsonl")
UNCACHED = Path("f0_sweep_odil_uncached.jsonl")
DEVITO = Path("f0_sweep_devito_na.jsonl")
SCALING = Path("strong_scaling.jsonl")
FIGURE = "fig4_bare_scaling.png"

TITLE_FS = 16
LABEL_FS = 14

EXCLUDE_COLS = exclude = [
    "wall",
    "walls",
    "omp",
    "ppw",
    "restart",
    "n_repeats",
    "rho",
]

COL = {"cached": "royalblue", "uncached": "seagreen", "na": "darkorange"}

LABELS = {
    "na": "Devito",
    "cached": "ODIL (caching)",
    "uncached": "ODIL (non-caching)",
}


def plot_memory_scaling(cached, uncached, devito, ax):
    """Plot the memory scaling behaviour of the 3 methodologies"""

    # plot peak RSS in GiB
    for df, colour, label in [
        (cached, COL["cached"], LABELS["cached"]),
        (uncached, COL["uncached"], LABELS["uncached"]),
        (devito, COL["na"], LABELS["na"]),
    ]:
        df = df.sort_values("ns")
        ax.loglog(df["ns"], df["peak_rss_GiB"], "o-", color=colour, label=label)

    # now for the reference complexities
    x0, y0 = cached["ns"].iloc[0], cached["peak_rss_GiB"].iloc[0]

    xs = np.linspace(
        cached["ns"].min(), max(cached["ns"].max(), uncached["ns"].max()) * 1.2, 200
    )

    # odil theoretical upper/lower memory bounds:
    # upper: O(n^2), lower: O(nlogn)
    lower = y0 * (xs * np.log(xs)) / (x0 * np.log(x0))
    upper = y0 * (xs / x0) ** 2

    ax.fill_between(xs, lower, upper, color="grey", alpha=0.08)
    ax.plot(xs, lower, ":", color="gray", alpha=0.5)
    ax.plot(xs, upper, ":", color="gray", alpha=0.5)

    # devito (single reference complexity, no bounds)
    xd0, yd0 = devito["ns"].iloc[-1], devito["peak_rss_GiB"].iloc[-1]
    ax.plot(xs, yd0 * (xs / xd0), ":", color="gray", alpha=0.5)

    x_mid = 1.2e4

    # annotate complexity lines
    ax.annotate(
        r"$O(n^2)$",
        xy=(x_mid, 30),
        xytext=(0, 0),
        textcoords="offset points",
        ha="right",
        fontsize=12,
        alpha=0.7,
        rotation=35,
    )
    ax.annotate(
        r"$O(n \log n)$",
        xy=(x_mid, 2),
        xytext=(15, -5),
        textcoords="offset points",
        ha="right",
        fontsize=12,
        alpha=0.7,
        rotation=25,
    )
    ax.annotate(
        r"$O(N_s)$",
        xy=(x_mid, 0.05),
        xytext=(10, 0),
        textcoords="offset points",
        ha="right",
        fontsize=12,
        alpha=0.7,
        rotation=25,
    )

    ax.set_xlabel(r"$N_s$ (spatial DOF)", fontsize=LABEL_FS)
    ax.set_ylabel("Peak RSS (GiB)", fontsize=LABEL_FS)
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_time_per_mode_scaling(cached, uncached, devito, ax):
    """Plot per mode / per step time vs complexities"""
    cached = cached.sort_values("ns")
    uncached = uncached.sort_values("ns")
    devito = devito.sort_values("ns")

    # setup = factorisation for cached, factodiation dominates solve for uncached
    per_mode_setup = cached["t_setup"] / cached["n_modes"]
    per_mode_solve = uncached["t_solve"] / (uncached["n_matvecs"] * uncached["n_modes"])
    kernel_per_step = devito["kernel_time"] / devito["nt"]

    ax.loglog(
        cached["ns"], per_mode_setup, "o-", color=COL["cached"], label=LABELS["cached"]
    )
    ax.loglog(
        uncached["ns"],
        per_mode_solve,
        "o-",
        color=COL["uncached"],
        label=LABELS["uncached"],
    )
    ax.loglog(devito["ns"], kernel_per_step, "o-", color=COL["na"], label=LABELS["na"])

    # odil theoretical bounds
    x0, y0 = cached["ns"].iloc[0], per_mode_setup.iloc[0]
    xs = np.linspace(
        cached["ns"].min(), max(cached["ns"].max(), uncached["ns"].max()) * 1.2, 200
    )

    lower = y0 * (xs / x0) ** 1.5  # O(n^3/2)
    upper = y0 * (xs / x0) ** 3  # O(n^3)

    ax.fill_between(xs, lower, upper, color="grey", alpha=0.08)
    ax.plot(xs, lower, ":", color="gray", alpha=0.5)
    ax.plot(xs, upper, ":", color="gray", alpha=0.5)

    # devito reference
    xd0, yd0 = devito["ns"].iloc[-1], kernel_per_step.iloc[-1]
    ax.plot(xs, yd0 * (xs / xd0), ":", color="gray", alpha=0.5)

    x_mid = 1e4

    # complexity annotations
    ax.annotate(
        r"$O(n^3)$",
        xy=(x_mid, 10),
        xytext=(-15, -5),
        textcoords="offset points",
        fontsize=12,
        alpha=0.7,
        rotation=25,
    )
    ax.annotate(
        r"$O(n^{\frac{3}{2}})$",
        xy=(x_mid, 0.2),
        xytext=(-15, -30),
        textcoords="offset points",
        fontsize=12,
        alpha=0.7,
        rotation=20,
    )
    ax.annotate(
        r"$O(N_s)$",
        xy=(x_mid, 1e-5),
        xytext=(-15, -10),
        textcoords="offset points",
        fontsize=12,
        alpha=0.7,
        rotation=10,
    )

    ax.set_xlabel(r"$N_s$ (spatial DOF)", fontsize=LABEL_FS)
    ax.set_ylabel("Time per mode / step (s)", fontsize=LABEL_FS)
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_raw_time(cached, uncached, devito, ax):
    """Raw wall clock vs Ns"""
    for df, colour, label in [
        (cached, COL["cached"], LABELS["cached"]),
        (uncached, COL["uncached"], LABELS["uncached"]),
        (devito, COL["na"], LABELS["na"]),
    ]:
        df = df.sort_values("dof")
        ax.plot(df["dof"], df["wall_mean"], "o-", color=colour, label=label)

    ax.set_xlabel(r"DOF ($N_x \times N_y \times N_t$)", fontsize=LABEL_FS)
    ax.set_ylabel("Wall clock time (s)", fontsize=LABEL_FS)
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_mem_scaling_f0(cached, uncached, devito, ax):
    """Plot Peak RSS versus f0"""
    for df, colour, label in [
        (cached, COL["cached"], LABELS["cached"]),
        (uncached, COL["uncached"], LABELS["uncached"]),
        (devito, COL["na"], LABELS["na"]),
    ]:
        df = df.sort_values("f0")
        ax.plot(df["f0"] / 1000.0, df["peak_rss_GiB"], "o-", color=colour)

    ax.set_xlabel(r"$f_0$ (kHz)", fontsize=LABEL_FS)
    ax.set_ylabel("Peak RSS (GiB)", fontsize=LABEL_FS)
    ax.grid(True, which="both", alpha=0.3)
    return ax


def main():
    """Helper to assemble full plot"""
    # load all
    cached = load_jsonl(which=CACHED, path=RESULTS_DIR)
    uncached = load_jsonl(which=UNCACHED, path=RESULTS_DIR)
    devito = load_jsonl(which=DEVITO, path=RESULTS_DIR)

    # clean up columns we don't care about
    # convert memory cols to GiB
    results = [cached, uncached, devito]
    for result in results:
        for col in EXCLUDE_COLS:
            if col in result.columns:
                result.drop(columns=col, inplace=True)
        result = bytes_to_gib(result)

    fig = plt.figure(figsize=(16, 12))
    grid = pltgs.GridSpec(2, 2, figure=fig)

    ax_time = fig.add_subplot(grid[0, 0])
    ax_raw = fig.add_subplot(grid[0, 1])
    ax_mem = fig.add_subplot(grid[1, 0])
    ax_f0 = fig.add_subplot(grid[1, 1])

    plot_time_per_mode_scaling(cached, uncached, devito, ax_time)
    plot_raw_time(cached, uncached, devito, ax_raw)
    plot_memory_scaling(cached, uncached, devito, ax_mem)
    plot_mem_scaling_f0(cached, uncached, devito, ax_f0)

    legend_handles = [
        Patch(facecolor=COL["cached"], label=LABELS["cached"]),
        Patch(facecolor=COL["uncached"], label=LABELS["uncached"]),
        Patch(facecolor=COL["na"], label=LABELS["na"]),
    ]

    fig.subplots_adjust(wspace=0.5)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=TITLE_FS,
        ncol=3,
    )

    # panel labels
    for ax, lab in zip([ax_raw, ax_mem, ax_time, ax_f0], "abcd"):
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

    fig.savefig(FIGURE, dpi=200, bbox_inches="tight", pad_inches=0.1)
    print("Results saved to", str(FIGURE))


if __name__ == "__main__":
    main()

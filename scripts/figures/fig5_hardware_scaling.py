from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from common import load_jsonl, bytes_to_gib

RESULTS_DIR = Path("results/performance/")
CACHED = Path("f0_sweep_odil_cached.jsonl")
UNCACHED = Path("f0_sweep_odil_uncached.jsonl")
DEVITO = Path("f0_sweep_devito_na.jsonl")
SCALING = Path("strong_scaling.jsonl")
FIGURE = "fig5_hardware_scaling.png"

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


def plot_strong_scaling(df: pd.DataFrame, ax):
    """Speedup vs ncores per solver/mode, against an linear reference"""
    for (_, mode), sub in df.groupby(["solver", "mode"]):
        sub = sub.sort_values("ncores")

        # compute speedup and plot
        base = sub["wall_mean"].iloc[0]
        speedup = base / sub["wall_mean"]
        ax.loglog(
            sub["ncores"], speedup, "o-", label=LABELS[str(mode)], color=COL[str(mode)]
        )

    # get core counts
    cores = sorted(df["ncores"].unique())

    # linear
    ideal = [c / cores[0] for c in cores]
    ax.plot(cores, ideal, "k--", alpha=0.4, marker=".", label="Ideal")

    for c, y in zip(cores, ideal):
        ax.annotate(
            f"{y:.0f}x",
            xy=(c, y),
            xytext=(-12, 5),
            textcoords="offset points",
            ha="center",
            fontsize=12,
            color="gray",
            alpha=0.8,
        )

    ax.set_xlabel("Number of cores", fontsize=LABEL_FS)
    ax.set_ylabel("Speedup", fontsize=LABEL_FS)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(cores, labels=[str(c) for c in cores])
    ax.set_yticks(cores, labels=[str(c) for c in cores])
    ax.minorticks_off()  # drop the auto-placed unlabeled minor log ticks
    ax.grid(True, which="both", alpha=0.3)
    return ax


def main():
    """Helper to assemble full plot"""
    scaling = load_jsonl(which=SCALING, path=RESULTS_DIR)

    # clean up columns we don't care about
    # convert memory cols to GiB
    scaling.drop(columns=EXCLUDE_COLS, inplace=True)
    bytes_to_gib(scaling)

    fig, ax_ss = plt.subplots(figsize=(8, 8))
    plot_strong_scaling(scaling, ax_ss)

    legend_handles = [
        Patch(facecolor=COL["cached"], label=LABELS["cached"]),
        Patch(facecolor=COL["uncached"], label=LABELS["uncached"]),
        Patch(facecolor=COL["na"], label=LABELS["na"]),
    ]

    fig.subplots_adjust(bottom=0.18)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=TITLE_FS,
        ncol=3,
    )

    fig.savefig(FIGURE, dpi=200, bbox_inches="tight", pad_inches=0.1)
    print("Results saved to", str(FIGURE))
    plt.show()


if __name__ == "__main__":
    main()

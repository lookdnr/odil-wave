from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

RESULTS_DIR = Path("results/performance/")
CACHED = Path("f0_sweep_odil_cached.jsonl")
UNCACHED = Path("f0_sweep_odil_uncached.jsonl")
DEVITO = Path("f0_sweep_devito_na.jsonl")
SCALING = Path("strong_scaling.jsonl")

TITLE_FS = 16
LABEL_FS = 14

EXCLUDE_COLS = exclude = [
    "wall",
    "walls",
    "omp",
    "n_matvecs",
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


def load_jsonl(which: Path, path: Path = RESULTS_DIR) -> pd.DataFrame:
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
        cached["ns"].min(), max(cached["ns"].max(), uncached["ns"].max()) * 2.5, 200
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

    for mem_gib, tag in [(128, "128 GiB node"), (512, "512 GiB node")]:
        ax.axhline(mem_gib, color="k", linestyle="--", alpha=0.3)
        ax.annotate(
            tag,
            xy=(xs[0], mem_gib),
            xytext=(70, 4),
            textcoords="offset points",
            ha="right",
            fontsize=12,
            alpha=0.5,
        )

    # annotate complexity lines
    ax.annotate(
        r"$O(n \log n)$",
        xy=(1e5, 50),
        xytext=(80, 0),
        textcoords="offset points",
        ha="right",
        fontsize=12,
        alpha=0.6,
    )
    ax.annotate(
        r"$O(n^2)$",
        xy=(1e5, 2.5e3),
        xytext=(60, 0),
        textcoords="offset points",
        ha="right",
        fontsize=12,
        alpha=0.6,
    )
    ax.annotate(
        r"$O(N_s)$",
        xy=(1e5, 1.03),
        xytext=(60, 0),
        textcoords="offset points",
        ha="right",
        fontsize=12,
        alpha=0.6,
    )

    ax.set_xlabel(r"$N_s$ (spatial DOF)", fontsize=LABEL_FS)
    ax.set_ylabel("Peak RSS (GiB)", fontsize=LABEL_FS)
    ax.grid(True, which="both", alpha=0.3)
    return ax


def main(results):
    """Helper to assemble full plot"""
    fig, (ax_ss, ax_mem) = plt.subplots(1, 2, figsize=(16, 7))
    plot_strong_scaling(scaling, ax_ss)
    plot_memory_scaling(cached, uncached, devito, ax_mem)

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

    plt.show()


if __name__ == "__main__":
    # load all
    cached = load_jsonl(which=CACHED)
    uncached = load_jsonl(which=UNCACHED)
    devito = load_jsonl(which=DEVITO)
    scaling = load_jsonl(which=SCALING)

    # clean up columns we don't care about
    # convert memory cols to GiB
    results = [cached, uncached, devito, scaling]
    for result in results:
        for col in EXCLUDE_COLS:
            if col in result.columns:
                result.drop(columns=col, inplace=True)
        result = bytes_to_gib(result)

    main(results)

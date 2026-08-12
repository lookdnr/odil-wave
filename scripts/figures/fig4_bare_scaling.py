from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt

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
    labels = {
        "na": "Devito",
        "cached": "ODIL (caching)",
        "uncached": "ODIL (non-caching)",
    }
    for (_, mode), sub in df.groupby(["solver", "mode"]):
        sub = sub.sort_values("ncores")

        # compute speedup and plot
        base = sub["wall_mean"].iloc[0]
        speedup = base / sub["wall_mean"]
        ax.loglog(sub["ncores"], speedup, "o-", label=labels[str(mode)])

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
    ax.legend(fontsize=LABEL_FS)
    ax.grid(True, which="both", alpha=0.3)
    return ax


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

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_strong_scaling(scaling, ax)
    plt.show()

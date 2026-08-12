from pathlib import Path
import pandas as pd
import json

RESULTS_DIR = Path("results/performance/")
CACHED = Path("f0_sweep_odil_cached.jsonl")
UNCACHED = Path("f0_sweep_odil_uncached.jsonl")
DEVITO = Path("f0_sweep_devito_na.jsonl")
SCALING = Path("strong_scaling.jsonl")

EXCLUDE_COLS = exclude = [
    "wall",
    "walls",
    "omp",
    "n_matvecs",
    "mode",
    "ppw",
    "restart",
    "n_repeats",
    "solver",
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
        result.drop(columns=EXCLUDE_COLS, inplace=True)
        result = bytes_to_gib(result)

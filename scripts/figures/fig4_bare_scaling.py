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
    "per_mode_bytes",
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


if __name__ == "__main__":
    # load all
    cached = load_jsonl(which=CACHED)
    uncached = load_jsonl(which=UNCACHED)
    devito = load_jsonl(which=DEVITO)
    scaling = load_jsonl(which=SCALING)

    # clean up columns we don't care about
    results = [cached, uncached, devito, scaling]
    for result in results:
        result.drop(columns=EXCLUDE_COLS, inplace=True)

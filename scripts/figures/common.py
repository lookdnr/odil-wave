import json
from pathlib import Path
import pandas as pd
import pickle


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
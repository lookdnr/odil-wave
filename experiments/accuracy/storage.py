import pickle
from pathlib import Path
from typing import List

from .measure_accuracy import AccuracyResult


def save(results: List[AccuracyResult], path):
    """Save an accuracy sweep (list of AccuracyResult objects)"""
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(results, f)


def load(path):
    """Load an accuracy sweep from pickled file"""

    with open(path, "rb") as f:
        return pickle.load(f)

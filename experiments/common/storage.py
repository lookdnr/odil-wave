import pickle
from pathlib import Path


def save(results, path):
    """Save a pickleable object to path"""
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(results, f)


def load(path):
    """Load an accuracy sweep from pickled file"""

    with open(path, "rb") as f:
        return pickle.load(f)

import numpy as np
from typing import Callable, Tuple, Dict


def repeat(fn: Callable, n: int) -> Tuple[Dict, Dict]:
    """Call fn() n times, return the last metrics dict + wall-time statistics"""
    runs = [fn() for _ in range(n)]
    walls = [r["wall"] for r in runs]
    return runs[-1], dict(
        wall_mean=float(np.mean(walls)),
        wall_std=float(np.std(walls, ddof=1)) if n > 1 else 0.0,
        wall_min=float(np.min(walls)),
        walls=walls,  # raw wall clock
        n_repeats=n,
    )

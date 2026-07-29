import math


def points_per_wavelength(max_spacing: float, f0: float, c: float) -> float:
    """Compute the PPW achieved at wavespeed c and frequency f0 for a given grid"""
    return (c / f0) / max_spacing


def nodes_for_ppw(length: float, f0: float, ppw: float, c: float) -> int:
    """Smallest node count over an axis of physical `length` that resolves at
    least `ppw` points per shortest wavelength (lambda = c / f0)
    """
    lam = c / f0
    return math.ceil(ppw * length / lam) + 1

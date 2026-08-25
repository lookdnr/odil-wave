import h5py
import numpy as np

def load_h5(path="../../alpha2D-TrueModel.h5") -> np.ndarray:
    """Load model from .h5 file. Defaults to proprietary Sonalis brain atlas"""
    with h5py.File(path, "r") as f:
        return f["data"][()]  # type: ignore
from dataclasses import dataclass
import numpy as np


# exp 0
@dataclass
class BaselineResult:
    method: str
    wall: float
    iters: int
    converged: bool
    message: str
    err: float


# exp 1


@dataclass
class ReceiverReport:
    index: int
    xy: np.ndarray  # (x, y)
    r: float  # distance to source
    err_odil: float  # windowed error, this receiver only
    err_dev: float


@dataclass
class AccuracyResult:
    metrics: dict
    traces: dict
    receivers: list[ReceiverReport]

    def trace(self, k: int, solver: str = "odil"):
        """(time, numerical, analytic, window mask) for kth receiver"""

        if solver not in ["odil", "dev"]:
            raise ValueError("solver must be 'odil' or 'dev', got", solver)
        s = solver

        tr, rec = self.traces, self.receivers[k]

        return dict(
            rec=rec,
            t=tr[f"t_{s}"],
            numerical=tr[f"d_{s}"][:, k],
            analytical=tr[f"ana_{s}"][:, k],
            mask=tr[f"mask_{s}"][:, k],
        )


# exp3a


@dataclass
class DispersionResult:
    ppw: float
    nx: int
    nt: int
    dt: float
    angles: dict  # angle: dict(radii, freqs, c_{method}, alpha_{method})

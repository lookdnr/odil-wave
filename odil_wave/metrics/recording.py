from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np
from scipy.stats import gmean
from scipy.optimize import OptimizeResult

from odil_wave.wavefield import Wavefield


class SolveResult(OptimizeResult):
    """End state of a solve: solution wavefield + full recording."""

    def __init__(self, wavefield, recorder, nit, success, message=""):
        last = recorder.outers[-1] if recorder.outers else None
        super().__init__(
            x=wavefield.flat_data,
            fun=0.5 * last.res**2 if last else np.nan,
            nit=nit,
            success=success,
            message=message,
            wavefield=wavefield,
            recorder=recorder,
        )


@dataclass
class InnerRecord:
    """History for one inner solve of the GN problem"""

    residual_history: List[float]  # full pr_norm curve
    true_relres: float  # ||Adu + r|| / ||r|| at exit
    converged: bool
    n_matvecs: int
    t_solve: float  # inner solve duration

    @property
    def iters(self):
        """Number of inner iterations"""
        return len(self.residual_history)

    @property
    def rho(self):
        """Estimated convergence factor: geometric mean of successive ratios"""
        r_arr = np.array(self.residual_history)
        ratios = r_arr[1:] / r_arr[:-1]
        return gmean(ratios)


@dataclass
class OuterRecord:
    """Record for one outer GN step"""

    res: float  # ||Au - s||
    grad_norm: float  # norm of gradient
    relres: float  # ||Au - s|| / ||s||
    inner: InnerRecord | None  # none for scipy


@dataclass
class SolveRecorder:
    """Dataclass for recording solve history"""

    meta: Dict = field(default_factory=dict)  # method, alpha, rtol, restart, ...
    outers: List[OuterRecord] = field(default_factory=list)

    def log(self, residuals: np.ndarray, grad=None) -> None:
        """Log norms for one outer solve"""
        r = float(np.linalg.norm(residuals))
        self.outers.append(
            OuterRecord(
                res=r,
                relres=r / self.meta["norm_s"],
                grad_norm=float(np.linalg.norm(grad)) if grad is not None else np.nan,
                inner=None,
            )
        )

    def log_inner(self, record: InnerRecord) -> None:
        self.outers[-1].inner = record

    def finalise(
        self, wavefield: Wavefield, nit: int, success: bool, message=""
    ) -> SolveResult:
        return SolveResult(wavefield, nit, success, message)

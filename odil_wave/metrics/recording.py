from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.stats import gmean
from scipy.optimize import OptimizeResult
import matplotlib.pyplot as plt

from odil_wave.wavefield import Wavefield


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
        norm_s = self.meta.get("norm_s")
        self.outers.append(
            OuterRecord(
                res=r,
                relres=r / norm_s if norm_s else np.nan,
                grad_norm=float(np.linalg.norm(grad)) if grad is not None else np.nan,
                inner=None,
            )
        )

    def log_inner(self, record: InnerRecord) -> None:
        self.outers[-1].inner = record

    def finalise(
        self, wavefield: Wavefield, nit: int, success: bool, message: str = ""
    ) -> SolveResult:
        return SolveResult(wavefield, self, nit, success, message)

    def save(self, path: str | Path) -> Path:
        """Write metadata + full outer/inner history to JSON"""
        self.meta.setdefault("saved_at", datetime.now().isoformat())
        path = Path(path).with_suffix(".json")
        payload = {"meta": self.meta, "outers": [asdict(o) for o in self.outers]}
        path.write_text(json.dumps(payload, indent=2, default=float))
        return path

    @classmethod
    def load(cls, path: str | Path) -> SolveRecorder:
        """Load metadata and outer/inner history from JSON payload"""
        data = json.loads(Path(path).read_text())
        outers = []

        for d in data["outers"]:
            inner = d.pop("inner")
            outers.append(
                OuterRecord(**d, inner=InnerRecord(**inner) if inner else None)
            )
        return cls(meta=data["meta"], outers=outers)


class SolveResult(OptimizeResult):
    """End state of a solve: solution wavefield + full recording."""

    def __init__(
        self,
        wavefield: Wavefield,
        recorder: SolveRecorder,
        nit: int,
        success: bool,
        message: str = "",
    ):
        last = recorder.outers[-1] if recorder.outers else None

        self.wf = wavefield

        super().__init__(
            x=wavefield.flat_data,
            fun=0.5 * last.res**2 if last else np.nan,
            nit=nit,
            success=success,
            message=message,
            wavefield=wavefield,
            recorder=recorder,
        )

    @property
    def solution(self) -> Wavefield:
        return self.wf

    def show_convergence(self, title: str = "Convergence"):
        """Plot outer convergence history"""
        outers = self["recorder"].outers

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        axs[0].semilogy([o.relres for o in outers], marker="o")
        axs[0].set(title="Relative residual", xlabel="Outer iteration")

        axs[1].semilogy([o.grad_norm for o in outers], marker="o")
        axs[1].set(title="Gradient norm", xlabel="Outer iteration")

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    def show_inner(self, title: str = "Inner solve history"):
        """Plot the inner solve history"""
        inners = [
            (k, o.inner) for k, o in enumerate(self["recorder"].outers) if o.inner
        ]

        if not inners:
            print("No inner solves recorded.")
            return None

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        for k, inner in inners:  # the plot the study runs on
            axs[0].semilogy(
                inner.residual_history, label=rf"outer {k} ($\rho$={inner.rho:.2f})"
            )

        axs[0].set(title="GMRES residual history", xlabel="Inner iteration")
        axs[0].legend()

        ks = [k for k, _ in inners]

        colours = ["tab:blue" if i.converged else "tab:orange" for _, i in inners]

        axs[1].bar(ks, [i.iters for _, i in inners], color=colours)
        axs[1].set(title="Inner iterations", xlabel="Outer iteration")

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    def save(self, path: str | Path, with_field: bool = False) -> Path:
        """Persist the recording to disk, optionally with the solution field"""
        out = self["recorder"].save(path)

        if with_field:
            np.savez_compressed(out.with_suffix(".npz"), u=self.x)
        return out

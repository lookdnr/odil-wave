from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict
import json
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
from scipy.stats import gmean
from scipy.optimize import OptimizeResult
import matplotlib.pyplot as plt

from odil_wave.wavefield import Wavefield


@dataclass
class InnerRecord:
    """History for one inner solve of the Gauss-Newton problem.
    That inner solve is Adu = -r, orchestrated by GMRES.

    Parameters
    ----------
    residual_history : list of float
        Residual norm curve over inner iterations.
    true_relres : float
        True relative residual ||Adu + r|| / ||r|| at exit.
    converged : bool
        Whether the inner solve met its convergence tolerance.
    n_matvecs : int
        Number of matrix-vector products used.
    t_solve : float
        Inner solve wall clock duration [s].
    """

    residual_history: List[float]  # full pr_norm curve
    true_relres: float  # ||Adu + r|| / ||r|| at exit
    converged: bool
    n_matvecs: int
    t_solve: float  # inner solve duration

    @property
    def iters(self):
        """int: number of inner iterations."""
        return len(self.residual_history)

    @property
    def rho(self):
        """float: Estimated convergence factor, computed as the geometric mean
        of successive residual ratios. NaN with a warning if fewer than 2
        iterations were recorded.
        """
        r_arr = np.array(self.residual_history)

        if len(r_arr) < 2:
            warnings.warn(
                "residual history has length of 1, geometric mean undefined.",
                RuntimeWarning,
            )
            return np.nan

        ratios = r_arr[1:] / r_arr[:-1]
        return gmean(ratios)


@dataclass
class OuterRecord:
    """Record for one outer Gauss Newton iteration.

    Parameters
    ----------
    res : float
        PDE residual norm ||Au - s||.
    grad_norm : float
        Gradient norm at this step.
    relres : float
        Relative residual ||Au - s|| / ||s||.
    inner : InnerRecord or None
        Inner solve history for this step, None for scipy optimisers.
    """

    res: float  # ||Au - s||
    grad_norm: float  # norm of gradient
    relres: float  # ||Au - s|| / ||s||
    inner: InnerRecord | None  # none for scipy


@dataclass
class SolveRecorder:
    """Recorder for the full outer/inner history of a Gauss Newton solve.

    Parameters
    ----------
    meta : dict, optional
        Run metadata (method, alpha, rtol, restart, ...).
    outers : list of OuterRecord, optional
        Per outer iteration records, appended to via `log`/`log_inner`.
    """

    meta: Dict = field(default_factory=dict)  # method, alpha, rtol, restart, ...
    outers: List[OuterRecord] = field(default_factory=list)

    def log(self, residuals: np.ndarray, grad=None) -> None:
        """Log residual and gradient norms for one outer iteration.

        Parameters
        ----------
        residuals : np.ndarray
            PDE residual at this outer iteration.
        grad : np.ndarray, optional
            Gradient at this outer iteration, logged as NaN if None.
        """
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
        """Attach an inner solve record to the most recent outer iteration.

        Parameters
        ----------
        record : InnerRecord
            Inner solve history to attach.
        """
        self.outers[-1].inner = record

    def finalise(
        self, wavefield: Wavefield, nit: int, success: bool, message: str = ""
    ) -> SolveResult:
        """Package the recorded history into a `SolveResult`.

        Parameters
        ----------
        wavefield : Wavefield
            Final solution wavefield.
        nit : int
            Number of outer iterations taken.
        success : bool
            Whether the solve converged.
        message : str, optional
            Solver termination message.

        Returns
        -------
        SolveResult
            The finalised solve result.
        """
        return SolveResult(wavefield, self, nit, success, message)

    def save(self, path: str | Path) -> Path:
        """Write metadata and full outer/inner history to JSON.

        Parameters
        ----------
        path : str or Path
            Output path; a ".json" suffix is enforced.

        Returns
        -------
        pathlib.Path
            Path the history was written to.
        """
        self.meta.setdefault("saved_at", datetime.now().isoformat())
        path = Path(path).with_suffix(".json")
        payload = {"meta": self.meta, "outers": [asdict(o) for o in self.outers]}
        path.write_text(json.dumps(payload, indent=2, default=float))
        return path

    @classmethod
    def load(cls, path: str | Path) -> SolveRecorder:
        """Load metadata and outer/inner history from a JSON payload.

        Parameters
        ----------
        path : str or Path
            Path to a JSON file previously written by `save`.

        Returns
        -------
        SolveRecorder
            Recorder reconstructed from the saved history.
        """
        data = json.loads(Path(path).read_text())
        outers = []

        for d in data["outers"]:
            inner = d.pop("inner")
            outers.append(
                OuterRecord(**d, inner=InnerRecord(**inner) if inner else None)
            )
        return cls(meta=data["meta"], outers=outers)


class SolveResult(OptimizeResult):
    """End state of a solve: solution wavefield plus full recording.

    Subclasses `scipy.optimize.OptimizeResult`, so standard fields (`x`,
    `fun`, `nit`, `success`, `message`) are populated alongside
    `wavefield` and `recorder`.

    Parameters
    ----------
    wavefield : Wavefield
        Final solution wavefield.
    recorder : SolveRecorder
        Full outer/inner solve history.
    nit : int
        Number of outer iterations taken.
    success : bool
        Whether the solve converged.
    message : str, optional
        Solver termination message.
    """

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
        """Wavefield: the solution wavefield"""
        return self.wf

    def show_convergence(self, title: str = "Convergence"):
        """Plot outer iteration convergence history.

        Parameters
        ----------
        title : str, optional
            Figure title.

        Returns
        -------
        matplotlib.figure.Figure
            Figure with two panels: relative residual and gradient norm
            vs outer iteration.
        """
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
        """Plot inner solve convergence history across outer iterations.

        Parameters
        ----------
        title : str, optional
            Figure title.

        Returns
        -------
        matplotlib.figure.Figure or None
            Figure with GMRES residual curves and iteration counts per
            outer step, None (with a printed message) if no inner solves
            were recorded.
        """
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
        """Persist the recording to disk, optionally with the solution field.

        Parameters
        ----------
        path : str or Path
            Output path for the recorder JSON.
        with_field : bool, optional
            If True, also save the solution array to a companion ".npz" file.

        Returns
        -------
        Path
            Path the recorder JSON was written to.
        """
        out = self["recorder"].save(path)

        if with_field:
            np.savez_compressed(out.with_suffix(".npz"), u=self.x)
        return out

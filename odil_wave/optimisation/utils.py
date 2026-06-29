import matplotlib.pyplot as plt
import numpy as np
from odil_wave import Wavefield
from odil_wave.loss.utils import LossTape

from scipy.optimize import OptimizeResult

from dataclasses import dataclass
from typing import List


def create_u0(u0: Wavefield | np.ndarray | None = None, N: int = -1) -> np.ndarray:
    """Utility for creating flattened u0"""
    if u0 is None:
        u0 = np.zeros(N)
    elif isinstance(u0, Wavefield):
        u0 = u0.flat_data
    elif isinstance(u0, np.ndarray):
        u0 = np.asarray(u0).ravel()
    else:
        raise TypeError(
            "arg `u0` must be one of Wavefield, np.ndarray, None, got" + f" {type(u0)}"
        )
    return u0


@dataclass
class InnerSolveInfo:
    """Diagnostics from one inner iterative solve."""

    itn: int  # iterations used
    normr: float | None  # ‖b − Ax‖ at exit; None for cg
    normar: float | None  # ‖Aᵀ(b − Ax)‖ at exit; None for cg
    converged: bool  # False if the solver hit maxiter


class OptimisationResult(OptimizeResult):
    """Wraps a completed optimisation run.

    Inherits from scipy.optimize.OptimizeResult (a dict with attribute access),
    so standard fields (x, fun, nit, success, message) work as expected.
    Extra fields: wavefield (Wavefield), tape (LossTape), inner_history (InnerHistory).
    """

    def __init__(
        self,
        wavefield: Wavefield,
        tape: LossTape,
        nit: int,
        success: bool,
        inner_history: List[InnerSolveInfo],
        message: str = "",
    ) -> None:
        loss_history = tape.history["loss"]
        super().__init__(
            x=wavefield.flat_data,
            fun=loss_history[-1] if loss_history else np.nan,
            nit=nit,
            success=success,
            message=message,
            wavefield=wavefield,
            tape=tape,
            inner_history=inner_history,
        )

    @property
    def solution(self) -> Wavefield:
        return self["wavefield"]

    @property
    def success(self) -> bool:
        return self["success"]

    @property
    def exit_reason(self) -> None:
        print(self["message"])

    def show_inner(self, title: str = "Inner Solve History"):
        history: list[InnerSolveInfo] = self["inner_history"]
        if not history:
            print("No inner solve history recorded.")
            return

        itns = [h.itn for h in history]
        normrs = [h.normr for h in history if h.normr is not None]
        normars = [h.normar for h in history if h.normar is not None]
        outer_iters = range(1, len(itns) + 1)

        n_plots = 1 + bool(normrs) + bool(normars)
        fig, axs = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
        if n_plots == 1:
            axs = [axs]

        colours = ["tab:orange" if not h.converged else "tab:blue" for h in history]
        axs[0].bar(outer_iters, itns, color=colours)
        axs[0].set_title("Inner iterations")
        axs[0].set_xlabel("Outer iteration")
        axs[0].set_ylabel("itn")

        ax_idx = 1
        if normrs:
            axs[ax_idx].semilogy(range(1, len(normrs) + 1), normrs, marker="o")
            axs[ax_idx].set_title(r"$\|J^\top J \delta u - J^\T r\|$ at exit")
            axs[ax_idx].set_xlabel("Outer iteration")
            ax_idx += 1

        if normars:
            axs[ax_idx].semilogy(range(1, len(normars) + 1), normars, marker="o")
            axs[ax_idx].set_title(r"$\|A^\top(J^\top J \delta u - J^\top r)\|$ at exit")
            axs[ax_idx].set_xlabel("Outer iteration")

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import scipy.optimize as scopt
import numpy as np


@dataclass
class LossTape:
    """Tape to store the loss history."""

    name: str = "Default LossTape"
    log_every: int = 5  # log interval
    history: dict = field(default_factory=lambda: {"loss": [], "pde_residuals": []})
    _norm_cache: dict = field(default_factory=lambda: {"pde": []})
    _result: scopt.OptimizeResult = field(init=False)  # store optimisation result

    def _norms(self, key: str, cache_key: str) -> list:
        """Return residual norms, computing only entries not already cached."""
        residuals = self.history[key]
        cache = self._norm_cache[cache_key]

        for r in residuals[len(cache) :]:
            cache.append(float(np.linalg.norm(r)))

        return cache

    def log(self, loss: float, residuals: np.ndarray) -> None:
        """Log the loss and residuals."""
        self.history["loss"].append(loss)
        self.history["pde_residuals"].append(residuals)

    def show(self, title: str = "Loss History"):
        assert len(self.history["loss"]) > 0, "No loss history to show."
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # compute residual norms (cached; only new entries recomputed)
        pde_norms = self._norms("pde_residuals", "pde")

        axs[0].semilogy(self.history["loss"])
        axs[0].set_title("Loss")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Loss Value")

        axs[1].semilogy(pde_norms)
        axs[1].set_title("PDE Residual Norms")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Residual Norm")

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    @property
    def result(self) -> scopt.OptimizeResult:
        return self._result

    @result.setter
    def result(self, value: scopt.OptimizeResult):
        self._result = value

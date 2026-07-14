from dataclasses import dataclass, field
from typing import Dict, Callable, ClassVar
import warnings
import numpy as np

from odil_wave.wavefield import Wavefield
from odil_wave.geometry import Sources, Receivers
from odil_wave.operator import WaveEquation
from .error import (
    residual,
    residual_l2,
    residual_linfty,
    relative_l2,
    relative_linfty,
    final_time_l2,
    l2_error_history,
    relative_pde_residual,
)
from .trace import trace_misfit, trace_misfit_norm


@dataclass
class ErrorReport:
    """Convenience object for collecting metrics"""

    # fields
    u: Wavefield | np.ndarray
    u_ref: Wavefield | np.ndarray | None = None
    A: WaveEquation | None = None
    sources: Sources | None = None
    receivers: Receivers | None = None

    # method library: name : func
    residual_methods: ClassVar[Dict[str, Callable]] = {
        "residual_field": residual,
        "residual_infty_norm": residual_linfty,
        "residual_l2_norm": residual_l2,
        "pde_residual_relative": relative_pde_residual,
    }

    relative_methods: ClassVar[Dict[str, Callable]] = {
        "relative_infty_norm": relative_linfty,
        "relative_l2_norm": relative_l2,
        "relative_l2_history": l2_error_history,
        "relative_l2_final_time": final_time_l2,
    }

    trace_methods: ClassVar[Dict[str, Callable]] = {
        "trace_misfit": trace_misfit,
        "norm_of_trace_misfit": trace_misfit_norm,
    }

    # convenience bools
    can_compute_relative: bool = field(init=False, default=False)
    can_compute_residuals: bool = field(init=False, default=False)
    can_compute_trace: bool = field(init=False, default=False)

    # data results
    results: Dict[str, object] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        # check what was passed
        if self.u_ref is not None:
            self.can_compute_relative = True

        if self.A is not None and self.sources is not None:
            self.can_compute_residuals = True

        if self.can_compute_relative and self.receivers is not None:
            self.can_compute_trace = True

        skipped = {
            "relative metrics (needs u_ref)": self.can_compute_relative,
            "residual metrics (needs A and sources)": self.can_compute_residuals,
            "trace metrics (needs u_ref and receivers)": self.can_compute_trace,
        }
        skipped = [msg for msg, capable in skipped.items() if not capable]

        if len(skipped) == 3:
            raise ValueError(
                "ErrorReport cannot compute anything with only `u`: "
                "pass u_ref for relative metrics, A and sources for residual "
                "metrics, and u_ref plus receivers for trace metrics."
            )

        if skipped:
            warnings.warn(
                "ErrorReport will skip: " + "; ".join(skipped) + ".",
                UserWarning,
                stacklevel=2,
            )

    def generate(self) -> Dict[str, object]:
        """Generate full error report"""

        # compute residual terms
        if self.can_compute_residuals:
            for name, method in self.residual_methods.items():
                self.results[name] = method(self.A, self.u, self.sources)

        if self.can_compute_relative:
            for name, method in self.relative_methods.items():
                self.results[name] = method(self.u, self.u_ref)

        if self.can_compute_trace:
            for name, method in self.trace_methods.items():
                self.results[name] = method(self.u, self.u_ref, self.receivers)

        return self.results

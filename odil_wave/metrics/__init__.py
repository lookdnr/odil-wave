"""Error, trace, and convergence metrics for evaluating and recording solves.

- `error`/ `trace` provide field and trace level error metrics.
- `recording` provides `SolveRecorder`/ `SolveResult` for logging and persisting
outer/ inner solve history for the Gauss Newton method.
- `ErrorReport` bundles a solve's metrics into one report.
"""

from .error import (
    relative_l2,
    final_time_l2,
    relative_linfty,
    l2_error_history,
    residual,
    residual_l2,
    residual_linfty,
    relative_pde_residual,
)
from .trace import (
    trace_misfit,
    trace_misfit_norm,
    trace_rel_l2,
    normalised_trace_rel_l2,
)
from .recording import SolveRecorder, InnerRecord, SolveResult
from .reporting import ErrorReport

__all__ = [
    "relative_l2",
    "final_time_l2",
    "relative_linfty",
    "l2_error_history",
    "residual",
    "residual_l2",
    "residual_linfty",
    "relative_pde_residual",
    "trace_misfit",
    "trace_misfit_norm",
    "trace_rel_l2",
    "normalised_trace_rel_l2",
    "SolveRecorder",
    "InnerRecord",
    "SolveResult",
    "ErrorReport",
]

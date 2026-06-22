from dataclasses import dataclass, field
import torch
from odil_wave.wavefield import Wavefield
from .temporal import FirstTimeDerivative, SecondTimeDerivative
from .spatial import Laplacian


@dataclass
class WaveEquation:
    """Discrete acoustic wave equation u_tt - c^2*lap(u) = f."""

    wavefield: Wavefield
    time_order: int = 2
    space_order: int = 2

    _ut_op: FirstTimeDerivative = field(init=False)
    _utt_op: SecondTimeDerivative = field(init=False)
    _lap: Laplacian = field(init=False)

    def __post_init__(self):
        self._ut_op = FirstTimeDerivative(self.wavefield, self.time_order)
        self._utt_op = SecondTimeDerivative(self.wavefield, self.time_order)
        self._lap = Laplacian(self.wavefield, self.space_order)

    def residual(self, amp: torch.Tensor, wsp: torch.Tensor, source: torch.Tensor):
        """u_tt - c^2*(u_xx + u_yy) - f, with IC mismatch enforced at t=0."""
        pass

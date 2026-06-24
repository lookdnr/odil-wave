from .base import Optimiser
from odil_wave.wavefield import Wavefield
from odil_wave.loss import DiscreteLoss


class GaussNewtonOptimiser(Optimiser):
    """Gauss Newton Optimiser. Approximates the Hessian by H ~ J^T J, where
    J is the Jacobian of the functional, then solves:

        - J^T J du = -J^t r for du
        - u_k+1 = u_k + du to update
    """

    def __init__(self, loss: DiscreteLoss) -> None:
        self.loss = loss  # loss function

    def minimise(self, u0: Wavefield, maxiter: int, ftol: float, gtol: float):
        pass

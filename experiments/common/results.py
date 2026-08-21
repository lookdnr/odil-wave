from dataclasses import dataclass


@dataclass
class BaselineResult:
    method: str
    wall: float
    iters: int
    converged: bool
    message: str
    err: float

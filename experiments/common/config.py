from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
    """Input config for the experiments"""

    model: str
    nx: int
    ny: int
    xmax: float
    xmin: float
    ymax: float
    ymin: float
    c_min: float
    c_max: float
    cfl_safety: float
    f0: float
    source_loc: tuple[float, float]
    time_order: int
    space_order: int
    solver: str  # "odil" or "stride"
    method: str = "paradiag"
    alpha: float = 1e-3

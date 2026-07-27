from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class RunConfig:
    """Input config for the experiments"""

    # grid
    nx: int
    ny: int
    xmax: float
    xmin: float
    ymax: float
    ymin: float
    c_min: float
    c_max: float
    cfl_safety: float

    # source
    f0: float
    source_loc: Tuple[float, float]
    recv_locs: Tuple[Tuple[float, float]]

    # discretisation
    time_order: int
    space_order: int

    # model
    model: str = "homogeneous"
    contrast: float | None = None
    centre: Tuple[float, float] | None = None
    radius: float | None = None
    interior_fill: float | None = None
    mask_skull: bool | None = None

    # solving
    solver: str = "odil"  # "odil" or "stride"
    method: str = "paradiag"
    alpha: float = 1e-3

    def __post_init__(self):
        assert self.model in ["homogeneous", "inclusion", "shepp-logan"]
        assert self.solver in ["odil", "stride"]
        assert self.method in ["paradiag", "gmres"]

    @property
    def model_kwargs(self) -> Dict:
        match self.model:
            case "homogeneous":
                kwargs = dict(background_c=self.c_min)
            case "inclusion":
                kwargs = dict(
                    background_c=self.c_min,
                    contrast=self.contrast,
                    centre=self.centre,
                    radius=self.radius,
                )
            case "shepp-logan":
                kwargs = dict(
                    background_c=self.c_min,
                    contrast=self.contrast,
                    interior_fill=self.interior_fill,
                    mask_skull=self.mask_skull,
                )
        return kwargs

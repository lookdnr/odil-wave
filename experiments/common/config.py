import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

from odil_wave.grid.utils import points_per_wavelength, nodes_for_ppw


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

    # discretisation
    time_order: int
    space_order: int

    # source
    f0: float
    source_loc: Tuple[float, float]
    n_recvs: int = 1
    recv_locs: Tuple[Tuple[float, float], ...] | None = None
    recv_mode: str = "custom"
    a_frac: float = 0.55
    b_frac: float = 0.7
    ring_centre: Tuple[float, float] = (0.0, 0.0)

    # time (optional)
    t_max: float | None = None

    # model
    model: str = "homogeneous"
    contrast: float | None = None
    centre: Tuple[float, float] | None = None
    radius: float | None = None
    interior_fill: float | None = None
    mask_skull: bool | None = None

    # solving
    method: str = "paradiag"
    alpha: float = 1e-3
    rtol: float = 1e-8
    caching: bool = True
    maxiter: int = 500

    def __post_init__(self):
        assert self.recv_mode in ["custom", "ring"]
        assert self.model in ["homogeneous", "inclusion", "shepp-logan"]
        assert self.method in ["paradiag", "gmres", "lbfgs"]

        if self.assert_ppw():
            print("PPW:", self.ppw)

    @property
    def ppw(self) -> float:
        """compute points per shortest wavelength"""
        dx = (self.xmax - self.xmin) / (self.nx - 1)
        dy = (self.ymax - self.ymin) / (self.ny - 1)
        return points_per_wavelength(max(dx, dy), self.f0, self.c_min)

    def assert_ppw(self, min_ppw: float = 10.0) -> bool:
        """raise if the grid resolves fewer than min_ppw points per wavelength"""

        if self.ppw < min_ppw:
            n_needed = nodes_for_ppw(
                self.xmax - self.xmin, self.f0, min_ppw, self.c_min
            )
            raise ValueError(
                f"under-resolved: ppw={self.ppw:.1f} < {min_ppw} "
                f"(f0={self.f0}). increase nx/ny to >= {n_needed}, or lower f0."
            )
        return True

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

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "RunConfig":
        with open(path) as f:
            data = json.load(f)

        for key in ("source_loc", "ring_centre", "centre"):
            if data.get(key) is not None:
                data[key] = tuple(data[key])
        if data.get("recv_locs") is not None:
            data["recv_locs"] = tuple(tuple(loc) for loc in data["recv_locs"])

        return cls(**data)

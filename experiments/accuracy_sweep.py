import numpy as np
from dataclasses import replace

# test harness
from common import RunConfig
from accuracy import measure_accuracy_repeated
from accuracy.storage import save

# 24 receivers on a fixed circle: radius 0.045, centred, in the 0.2 m domain
_theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
RECV_LOCS = tuple((0.10 + 0.045 * np.cos(a), 0.10 + 0.045 * np.sin(a)) for a in _theta)

BASE = RunConfig(
    nx=70,
    ny=70,  # swept below these vals
    xmin=0.0,
    xmax=0.2,
    ymin=0.0,
    ymax=0.2,  # 20 cm
    c_min=1500.0,
    c_max=1500.0,  # homogeneous water
    cfl_safety=0.75,  # generous for order 6 in space
    t_max=1.2e-4,  # short horizon
    time_order=2,
    space_order=6,
    f0=50e3,  # 50khz
    source_loc=(0.10, 0.10),  # centred source
    recv_mode="custom",
    n_recvs=24,
    recv_locs=RECV_LOCS,  # centred ring
    model="homogeneous",
    method="paradiag",
    alpha=1e-3,  # rtol defaults to 1e-8
)

# create configs
nxs = [80, 100, 120, 140, 160, 180, 200, 250, 300]
configs = [replace(BASE, nx=n, ny=n) for n in nxs]

if __name__ == "__main__":
    # run and save sweep data and config
    results = []
    for cfg in configs:
        results.append(measure_accuracy_repeated(cfg))
        path = "results/accuracy/sweep_50khz"
        save(results, path=path + ".pkl")  # after each config
        cfg.to_json(path=path + f"_nx{cfg.nx}.json")

import matplotlib.pyplot as plt
import sys
from pathlib import Path

from odil_wave.geometry import AcquisitionGeometry

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

from common import (  # type: ignore
    build_grid,
    build_model,
    build_source,
    build_receivers,
)
from exp1_accuracy_sweep import BASE as HOMOG  # type: ignore
from exp2_performance_scaling import BASE as INCL  # type: ignore
from exp3_dispersion import BASE as RAYS  # type: ignore
from exp4_cfl_sweep import SL_BASE as SL  # type: ignore


def build_components(cfg):
    g = build_grid(cfg)
    m = build_model(cfg, g)
    s = build_source(cfg, g)
    r = build_receivers(cfg, g)
    return m, s, r


def main():
    """Plot examples of each model"""

    # build problem components from used configurations
    m_homog, s_homog, r_homog = build_components(HOMOG)
    m_incl, s_incl, r_incl = build_components(INCL)
    m_sl, s_sl, r_sl = build_components(SL)
    m_rays, s_rays, r_rays = build_components(RAYS)

    # build acqusition geometries
    geom_homog = AcquisitionGeometry(s_homog, r_homog)
    geom_incl = AcquisitionGeometry(s_incl, r_incl)
    geom_sl = AcquisitionGeometry(s_sl, r_sl)
    geom_rays = AcquisitionGeometry(s_rays, r_rays)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    geoms = [geom_homog, geom_incl, geom_sl, geom_rays]
    models = [m_homog, m_incl, m_sl, m_rays]
    titles = [
        "Homogeneous\nHalo geometry",
        "Inclusion",
        "Shepp Logan Phantom",
        "Homogeneous\nLinear geometry",
    ]

    for i, ax in enumerate(axs.ravel()):
        geoms[i].show(models[i], ax, cbar=False)
        ax.get_legend().remove()
        ax.set_title(titles[i])

        # remove cluttering y and x labels
        if i in [1, 3]:
            ax.set_ylabel("")

        if i in [0, 1]:
            ax.set_xlabel("")

    fig.tight_layout(h_pad=0.1, w_pad=0.6)
    plt.show()


if __name__ == "__main__":
    main()

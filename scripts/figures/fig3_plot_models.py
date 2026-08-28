import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
from pathlib import Path
import cmcrameri.cm as cmc

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

FIGURE = "fig3_models.png"


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

    fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)

    geoms = [geom_homog, geom_incl, geom_sl, geom_rays]
    models = [m_homog, m_incl, m_sl, m_rays]
    titles = [
        "Homogeneous\nHalo geometry",
        "Inclusion",
        "Shepp Logan Phantom",
        "Homogeneous\nRay geometry",
    ]

    # compute bounds for global cbar
    vmin = min(m.c_min for m in models)
    vmax = max(m.c_max for m in models)

    for i, ax in enumerate(axs.ravel()):
        geoms[i].show(
            models[i],
            ax,
            cbar=False,
            vmin=vmin,
            vmax=vmax,
            cmap=cmc.batlow,  # type: ignore
        )
        ax.get_legend().remove()
        ax.set_title(titles[i])

        # remove cluttering y and x labels
        if i in [1, 3]:
            ax.set_ylabel("")

        if i in [0, 1]:
            ax.set_xlabel("")

    im = axs.ravel()[0].images[0]
    cbar = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.02)
    cbar.set_label(r"c ($ms^{-1}$)", fontsize=16)

    # configure ticks
    ticks = [1500, 1825, 2150, 2475, 2800]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(tick) for tick in ticks])

    legend_handles = [
        Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            markersize=20,
            color="orangered",
            markeredgecolor="r",
            label="Source",
        ),
        Line2D(
            [],
            [],
            marker="v",
            color="lime",
            linestyle="none",
            markersize=15,
            markeredgewidth=1.0,
            markeredgecolor="k",
            fillstyle="none",
            label="Receiver",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
        fontsize=16,
        ncol=5,
    )

    for ax, label in zip(axs.ravel(), "abcd"):
        ax.text(
            -0.10,
            1.15,
            f"({label})",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
            fontsize=14,
        )

    return fig


if __name__ == "__main__":
    fig = main()
    plt.savefig(FIGURE, dpi=200)

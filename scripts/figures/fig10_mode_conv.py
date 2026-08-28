import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

from fig_utils import (
    load_field_wavefield,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

from exp4_cfl_sweep import SL_BASE  # type: ignore

PREFIX = "results/wave/cfl_sweep_sl_field"
CFL_VALUES = [0.7, 1.0, 1.3, 3.0, 5.0, 10.0, 17.2]
CFL_LIMIT = 1.0

FIGURE = "fig10_mode_conv.png"

T_START_US = 20.0  # src injection period

LABEL_FS = 14
TITLE_FS = 16


def mode_convergence_metrics(wf):
    """Cosine sim and per step growth of the field"""
    # extract field
    U = wf.U

    # compute norms
    norms = np.linalg.norm(U, axis=1)

    # filter to defined region
    # norms if > 0 else 1
    safe = np.where(norms > 0, norms, 1.0)

    # normalise
    U_hat = U / safe[:, None]

    # compute cosine sim between frames
    # this form is basically vectorised dot prod
    cos_sim = np.abs(np.sum(U_hat[:-1] * U_hat[1:,], axis=1))

    # compute growth rate between frames
    growth = norms[1:] / safe[:-1]
    return dict(t=wf.grid.t, cos_sim=cos_sim, growth=growth, norms=norms)


def plot_mode_convergence(figsize=(12, 6), growth_ylim=(0.9, 1.6)):
    """Plot cosine sim and growth per step for each CFL"""

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    for cfl in CFL_VALUES:
        # load field, compute metrics
        wf = load_field_wavefield(SL_BASE, PREFIX, cfl, "odil")
        m = mode_convergence_metrics(wf)

        # extract time, convert to mu s
        t_us = m["t"][1:] * 1e6

        # filter injection period
        early = t_us >= T_START_US
        live = early & (m["norms"][:-1] > 0)

        axs[0].plot(t_us[early], m["cos_sim"][early], label=str(cfl))
        axs[1].plot(t_us[live], m["growth"][live], label=str(cfl))

    # formatting
    axs[0].axhline(1.0, color="k", lw=1, ls=":")
    axs[0].set_ylim(0.0, 1.05)
    axs[0].set_xlabel(r"$t$ ($\mu$s)", fontsize=LABEL_FS)
    axs[0].set_ylabel(r"$s(t_i)$")
    axs[0].set_title("Cosine similarity", fontsize=TITLE_FS)

    axs[1].axhline(1.0, color="k", lw=0.8, ls=":")
    axs[1].set_ylim(*growth_ylim)
    axs[1].set_xlabel(r"$t$ ($\mu$s)", fontsize=LABEL_FS)
    axs[1].set_ylabel(r"$r(t_i)$")
    axs[1].set_title("Per step growth", fontsize=TITLE_FS)

    for ax, lab in zip(axs, "ab"):
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
        ax.text(
            -0.13,
            1.1,
            f"({lab})",
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=LABEL_FS,
            va="top",
            ha="left",
        )

    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="CFL",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
        ncol=len(CFL_VALUES),
        fontsize=LABEL_FS,
        title_fontsize=TITLE_FS,
    )
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    fig = plot_mode_convergence()
    fig.savefig(FIGURE, dpi=200, bbox_inches="tight", pad_inches=0.02)

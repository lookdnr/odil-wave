import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from fig_utils import load_pkl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

from wave_specific import analytical_phase_velocity # type:ignore
from exp3_dispersion import ANGLES # type: ignore


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "wave"

FIGURE = "fig6_dispersion.png"

TITLE_FS = 16
LABEL_FS = 14

COL = {"odil": "royalblue", "devito": "darkorange"}

DOMAIN_WIDTH=0.4

    
def plot_dispersion_once(ax, results, angle, c, domain_width=DOMAIN_WIDTH, f0=50e3):
    """Plot single phase velocity error panel, evaluated at f0"""
    ppws, err_odil, err_dev, err_theory = [], [], [], []
    for r in results:
        d = r.angles[angle]
        freqs = np.array(d["freqs"])
        h = domain_width / (r.nx - 1) # grid spacing for this ppw point
        idx = np.argmin(np.abs(freqs - f0)) # nearest measured bin to f0

        v_theory = analytical_phase_velocity(c / (freqs[idx] * h), angle, c, r.dt, h)

        # compute phase v errors at 50Khz
        ppws.append(r.ppw)
        err_odil.append(100 * (d["v_odil"][idx] / c - 1))
        err_dev.append(100 * (d["v_dev"][idx] / c - 1))
        err_theory.append(100 * (v_theory / c - 1))

    ax.plot(ppws, err_odil, "o-", color=COL["odil"])
    ax.plot(ppws, err_dev, "s-", color=COL["devito"])
    ax.plot(ppws, err_theory, "k--")
    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax.set(ylabel="Phase velocity error (%)", title=rf"$\theta={angle}^{{\circ}}$")


def plot_dispersion_summary(results, angles, c=1500.0, figsize=(11, 9)):
    """Plot phase velocity error for each result"""
    fig, axs = plt.subplots(2, 2, figsize=figsize, sharey=True)

    k = 0
    for row in range(2):
        for col in range(2):
            angle = angles[k]
            k += 1
            a = axs[row, col]

            plot_dispersion_once(a, results, angle, c)
            a.set_title(a.get_title(), fontsize=TITLE_FS)

            if row == 1:
                a.set_xlabel("PPW")
            else:
                a.set_xlabel("")
                a.tick_params(labelbottom=False)

            if col == 0:
                a.set_ylabel("Phase velocity error (%)")
            else:
                a.set_ylabel("")
                a.tick_params(labelleft=False)

    # globasl legend
    legend_handles = [
        Line2D([0], [0], color=COL["odil"], marker="o", label="ODIL"),
        Line2D([0], [0], color=COL["devito"], marker="s", label="Devito"),
        Line2D([0], [0], color="k", linestyle="--", label="Analytical"),
    ]
    
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
        fontsize=TITLE_FS,
        ncol=3,
    )

    for ax, lab in zip(axs.flat, "abcd"):
        ax.text(-0.12, 1.05, f"({lab})", transform=ax.transAxes, va="top", ha="left",
                 fontweight="bold", fontsize=14)

    return fig


if __name__ == "__main__":

    results = load_pkl(RESULTS_DIR / "dispersion.pkl")

    fig, ax = plt.subplots(figsize=(5 * len(ANGLES), 6))
    fig = plot_dispersion_summary(results, ANGLES)

    fig.savefig(FIGURE, dpi=200, bbox_inches="tight")

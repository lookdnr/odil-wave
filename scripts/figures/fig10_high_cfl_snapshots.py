import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import sys
from pathlib import Path

from fig_utils import (
    load_field_wavefield,
    reshape,
    add_model_contours,
    build_fine_model,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

from exp4_cfl_sweep import SL_BASE  # type: ignore

PREFIX = "results/wave/cfl_sweep_sl_field"
TIME_LEVELS = [15, 50, 80]
CFL_VALUES = [3.0, 6.0, 9.2]

FIGURE = "fig10_high_cfl.png"


def plot_high_cfl(fig_w=10.0, fig_h=9.0, cbar_frac=0.06):
    """Plot ODIL snapshots at high CFL numbers, one row per CFL value."""
    ncols = len(TIME_LEVELS)
    ncols_total = ncols + 1

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        len(CFL_VALUES),
        ncols_total,
        figure=fig,
        width_ratios=[1.0] * ncols + [cbar_frac],
        hspace=0.15,
        wspace=0.05,
    )

    model = build_fine_model(SL_BASE)

    for i, cfl in enumerate(CFL_VALUES):
        wf = load_field_wavefield(SL_BASE, PREFIX, cfl, solver="odil")
        amp, extent, t = reshape(wf)
        (xmin, xmax), (ymin, ymax) = extent

        frames = []
        for t_us in TIME_LEVELS:
            idx = int(np.argmin(np.abs(t - t_us * 1e-6)))
            frames.append(amp[idx])
        shared_peak = max(np.abs(f).max() for f in frames) or 1.0

        im = None
        for j, (t_us, frame) in enumerate(zip(TIME_LEVELS, frames)):
            ax = fig.add_subplot(gs[i, j])
            im = ax.imshow(
                frame.T,
                origin="lower",
                extent=(xmin, xmax, ymin, ymax),
                cmap="RdBu_r",
                vmin=-shared_peak,
                vmax=shared_peak,
            )
            add_model_contours(ax, model)
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title(rf"{t_us:.0f}$\mu$s", fontsize=14)
            if j == 0:
                ax.set_ylabel(
                    f"CFL = {cfl:g}", fontsize=14, rotation=0, ha="right", va="center"
                )

        cax = fig.add_subplot(gs[i, ncols])
        fig.colorbar(im, cax=cax, label="Amplitude")  # type: ignore

    return fig


if __name__ == "__main__":
    fig = plot_high_cfl()
    fig.savefig(FIGURE, dpi=500, bbox_inches="tight", pad_inches=0.02)

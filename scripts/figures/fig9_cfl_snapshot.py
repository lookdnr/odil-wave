import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from fig_utils import (
    load_field_wavefield,
    add_model_contours,
    build_fine_model,
    reshape,
)
from exp4_cfl_sweep import SL_BASE  # type: ignore

PREFIX = "results/wave/cfl_sweep_sl_field"
TIME_LEVELS = [15, 50, 80, 110]
CFL_LOW, CFL_HIGH = 0.7, 1.3
CFL_UNSTABLE = [3.0, 5.0, 10.0, 17.2]
SNAPSHOT_TIME = 300  # us, for the beyond-limit row

FIGURE = "fig9_cfl_snapshots.png"


def _draw(ax, frame, extent, model):
    (xmin, xmax), (ymin, ymax) = extent

    peak = np.abs(frame).max()

    ax.imshow(
        frame.T,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        cmap="RdBu_r",
        vmin=-peak,
        vmax=peak,
    )
    add_model_contours(ax, model)
    ax.set_xticks([])
    ax.set_yticks([])
    return peak


def plot_cfl_snapshot(gap_frac=0.1, label_frac=0.1, fig_w=11.0):
    """ODIL vs Devito at a stable and a marginal CFL, plus ODIL alone
    beyond the CFL limit at a single late time."""
    ncols = len(TIME_LEVELS)

    row_specs = [
        ("label", f"CFL = {CFL_LOW}"),
        ("field", CFL_LOW, "odil", "ODIL", True),
        ("field", CFL_LOW, "devito", "Devito", False),
        None,
        ("label", f"CFL = {CFL_HIGH}"),
        ("field", CFL_HIGH, "odil", "ODIL", True),
        ("field", CFL_HIGH, "devito", "Devito", False),
        None,
        ("label", rf"ODIL beyond the CFL limit ({SNAPSHOT_TIME}$\mathbf{{\mu}}$s)"),
        None,
        ("cfls", SNAPSHOT_TIME, CFL_UNSTABLE),
    ]

    height_ratios = []
    for spec in row_specs:
        if spec is None:
            height_ratios.append(gap_frac)
        elif spec[0] == "label":
            height_ratios.append(label_frac)
        else:
            height_ratios.append(1.0)

    fig_h = fig_w * sum(height_ratios) / ncols
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        len(row_specs),
        ncols,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.1,
        wspace=0.01,
    )

    model = build_fine_model(SL_BASE)

    for i, spec in enumerate(row_specs):
        if spec is None:
            continue

        if spec[0] == "label":
            ax = fig.add_subplot(gs[i, :])

            ax.axis("off")

            ax.text(
                0.5,
                0.5,
                spec[1],
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

            continue

        if spec[0] == "cfls":
            _, t_us, cfls = spec
            for j, cfl in enumerate(cfls):

                wf = load_field_wavefield(SL_BASE, PREFIX, cfl, "odil")
                amp, extent, t = reshape(wf)
                idx = int(np.argmin(np.abs(t - t_us * 1e-6)))

                ax = fig.add_subplot(gs[i, j])

                peak = _draw(ax, amp[idx], extent, model)

                ax.set_title(f"CFL = {cfl:g}", fontsize=14)

                ax.text(
                    0.03,
                    0.97,
                    f"Peak={peak:.1e}",
                    transform=ax.transAxes,
                    fontsize=12,
                    va="top",
                    ha="left",
                )

                if j == 0:
                    ax.set_ylabel(
                        "ODIL", fontsize=14, rotation=0, ha="right", va="center"
                    )

            continue

        _, cfl, solver, row_label, show_titles = spec
        wf = load_field_wavefield(SL_BASE, PREFIX, cfl, solver)
        amp, extent, t = reshape(wf)

        for j, t_us in enumerate(TIME_LEVELS):
            idx = int(np.argmin(np.abs(t - t_us * 1e-6)))
            ax = fig.add_subplot(gs[i, j])

            peak = _draw(ax, amp[idx], extent, model)

            if show_titles:
                ax.set_title(rf"{t_us:.0f}$\mu$s", fontsize=14)
            if j == 0:
                ax.set_ylabel(
                    row_label, fontsize=14, rotation=0, ha="right", va="center"
                )

            ax.text(
                0.03,
                0.97,
                f"Peak={peak:.1e}",
                transform=ax.transAxes,
                fontsize=12,
                va="top",
                ha="left",
            )

    return fig


if __name__ == "__main__":
    fig = plot_cfl_snapshot()
    fig.savefig(FIGURE, dpi=200, bbox_inches="tight")

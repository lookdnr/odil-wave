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

FIGURE = "fig9_cfl_snapshots.png"


def plot_cfl_snapshot(gap_frac=0.1, label_frac=0.1):
    """Plot panel of ODIL vs Devito snapshots at specific time levels and CFL numbers"""

    # set up panel
    ncols = len(TIME_LEVELS)

    # 5 rows, 2 each for devito and odil at specified cfls, 1 for odil residual
    row_specs = [
        ("label", f"CFL = {CFL_LOW}"),
        ("ODIL", CFL_LOW),
        ("Devito", CFL_LOW),
        None,
        ("label", f"CFL = {CFL_HIGH}"),
        ("ODIL", CFL_HIGH),
        ("Devito", CFL_HIGH),
        None,
        ("label", r"Relative ODIL residual between CFL regimes"),
        ("residual", None),
        ("colorbar", None),
    ]

    height_ratios = []
    for spec in row_specs:
        if spec is None:
            height_ratios.append(gap_frac)
        elif spec[0] in ("label", "colorbar"):
            height_ratios.append(label_frac)
        else:
            height_ratios.append(1.0)

    nrows = len(row_specs)

    fig = plt.figure(figsize=(11, 13))
    gs = GridSpec(
        nrows, ncols, figure=fig, height_ratios=height_ratios, hspace=0.1, wspace=0.01
    )

    # build fine model to show SL phantom contour
    model = build_fine_model(SL_BASE)

    for i, spec in enumerate(row_specs):
        if spec is None:  # skip centre
            continue

        # top level labels above rows
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

        # residual plot of odil at high and low cfl
        elif spec[0] == "residual":

            # extract and reshape wf
            amp_low_wf = load_field_wavefield(SL_BASE, PREFIX, CFL_LOW, "odil")
            amp_low, extent_low, t_low = reshape(amp_low_wf)
            amp_high_wf = load_field_wavefield(SL_BASE, PREFIX, CFL_HIGH, "odil")
            amp_high, _, t_high = reshape(amp_high_wf)
            (xmin, xmax), (ymin, ymax) = extent_low

            # compute relative residuals
            residuals = []
            for t_us in TIME_LEVELS:
                # get index of same time step
                idx_low = int(np.argmin(np.abs(t_low - t_us * 1e-6)))
                idx_high = int(np.argmin(np.abs(t_high - t_us * 1e-6)))

                frame_low = amp_low[idx_low]
                frame_high = amp_high[idx_high]
                reference_peak = np.abs(frame_low).max() or 1.0

                residuals.append(100 * (frame_high - frame_low) / reference_peak)

            shared_peak = max(np.abs(r).max() for r in residuals) or 1.0

            residual_axes, last_im = [], None
            for j, res in enumerate(residuals):
                ax = fig.add_subplot(gs[i, j])
                last_im = ax.imshow(
                    res.T,
                    origin="lower",
                    extent=(xmin, xmax, ymin, ymax),
                    cmap="RdBu_r",
                    vmin=-shared_peak,
                    vmax=shared_peak,
                )
                add_model_contours(ax, model)
                ax.set_xticks([])
                ax.set_yticks([])
                residual_axes.append(ax)

                mean_abs = np.mean(np.abs(res))
                ax.text(
                    0.03,
                    0.97,
                    f"Mean={mean_abs:.2f}%",
                    transform=ax.transAxes,
                    fontsize=12,
                    color="black",
                    va="top",
                    ha="left",
                )

            continue

        elif spec[0] == "colorbar":
            mid = ncols // 2
            cax = fig.add_subplot(
                gs[i, mid - 1 : mid + 1] if ncols % 2 == 0 else gs[i, mid]
            )
            fig.colorbar(
                last_im,  # type: ignore
                cax=cax,
                orientation="horizontal",
                pad=1.0,  # type: ignore
                label="Relative field difference (%)",
            )
            continue

        # load wavefield from spec
        solver_label, cfl = spec
        solver = "odil" if solver_label == "ODIL" else "devito"
        wf = load_field_wavefield(SL_BASE, PREFIX, cfl, solver)

        # reshape for plotting
        amp, extent, t = reshape(wf)
        (xmin, xmax), (ymin, ymax) = extent

        # plot at time intervals, adding a subplot for each
        for j, t_us in enumerate(TIME_LEVELS):
            idx = int(np.argmin(np.abs(t - t_us * 1e-6)))
            frame = amp[idx]
            peak = np.abs(frame).max()
            ax = fig.add_subplot(gs[i, j])

            # plot and add SL phantom contour
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

            if i in (1, 5):
                ax.set_title(rf"{t_us:.0f}$\mu$s", fontsize=14)

            if j == 0:
                ax.set_ylabel(
                    solver_label, fontsize=14, rotation=0, ha="right", va="center"
                )

    return fig


if __name__ == "__main__":
    fig = plot_cfl_snapshot()
    fig.savefig(FIGURE, dpi=600)

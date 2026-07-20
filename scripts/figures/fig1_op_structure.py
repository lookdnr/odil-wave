"""
Figure 1: a plot demonstrating the sparsity of the structure of the
operators which form the global wave equation operator matrix
"""

import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from odil_wave import Grid, HomogeneousModel, Wavefield, WaveEquation

OUTFILE = "fig1_op_structure.png"

NX, NY = 5, 5


def assemble_A(Dtt, C2L, nt, ns, dt2):
    """Interior wave-equation operator A = dt2 * (Dtt (x) I_ns - I_nt (x) C2L).

    This is the *interior* PDE discretisation only.
    """
    I_ns = sp.identity(ns, format="csr")
    I_nt = sp.identity(nt, format="csr")
    return dt2 * (sp.kron(Dtt, I_ns) - sp.kron(I_nt, C2L))


def add_ij_axes(ax, loc=(0.1, -0.4), length=0.4, color="black", fontsize=14):
    x0, y0 = loc
    arrow_kw = dict(arrowstyle="-|>", color=color, lw=1.5, shrinkA=0, shrinkB=0)

    a1 = ax.annotate(
        "",
        xy=(x0 + length, y0),
        xytext=(x0, y0),
        xycoords="axes fraction",
        arrowprops=arrow_kw,
    )
    a2 = ax.annotate(
        "",
        xy=(x0, y0 - length),
        xytext=(x0, y0),
        xycoords="axes fraction",
        arrowprops=arrow_kw,
    )
    t1 = ax.text(
        x0 + length + 0.02,
        y0,
        "$i$",
        transform=ax.transAxes,
        va="center",
        ha="left",
        fontsize=fontsize,
    )
    t2 = ax.text(
        x0,
        y0 - length - 0.03,
        "$j$",
        transform=ax.transAxes,
        va="top",
        ha="center",
        fontsize=fontsize,
    )

    for artist in (a1, a2, t1, t2):
        artist.set_in_layout(False)


def annotate_bttb(ax, ns, block_idx=5, color="crimson"):
    """Highlight one (B2, B1, B0) block-triplet on a spy(A) panel and show
    it repeating elsewhere along the diagonal to show BTTB structure
    """
    i = block_idx

    def block_rect(row, col, **kw):
        return mpatches.Rectangle(
            (col * ns - 0.5, row * ns - 0.5),
            ns,
            ns,
            fill=False,
            linewidth=1.3,
            zorder=5,
            **kw,
        )

    # first triplet: sub-, main-, super-diagonal blocks at block-row i
    for col, style in [
        (i - 1, dict(linestyle=":")),
        (i, dict()),
        (i + 1, dict(linestyle=":")),
    ]:
        ax.add_patch(block_rect(i, col, edgecolor=color, **style))

    # a second instance of the same triplet, further along the diagonal
    i2 = i + 5
    for col, style in [
        (i2 - 1, dict(linestyle=":")),
        (i2, dict()),
        (i2 + 1, dict(linestyle=":")),
    ]:
        ax.add_patch(block_rect(i2, col, edgecolor=color, **style))

    ax.text(
        350,
        130,
        r"Toeplitz blocks",
        color=color,
        ha="center",
        va="top",
        fontsize=12,
        clip_on=False,
    )

    ax.annotate(
        "",
        xy=(180, 140),
        xytext=(300, 140),
        color="crimson",
        fontsize=12,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
        annotation_clip=False,
    )

    ax.annotate(
        "",
        xy=(290, 250),
        xytext=(300, 140),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
        annotation_clip=False,
    )


def main():
    # construct a simple problem
    grid = Grid(nx=NX, ny=NY)
    model = HomogeneousModel(grid)
    wavefield = Wavefield(grid)
    we = WaveEquation(
        wavefield, model, time_order=2, space_order=2, bc_angles=(0.0, 60.0)
    )

    ns = we.nx * we.ny

    # extract the operators
    lapl = we._lap_op
    temporal = we._utt_op

    Dxx = lapl.Dxx[:, 1:-1]  # remove ghost nodes for sake of plot
    Dyy = lapl.Dyy[:, 1:-1]
    Dtt = temporal.Dtt

    # assemble the full global operator
    A = assemble_A(Dtt, we.C2L, we.nt, ns, we.dt2)

    # plot sparse structure
    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 2])

    ax_dxx = fig.add_subplot(gs[0, 0:2])
    ax_dyy = fig.add_subplot(gs[0, 2:4])
    ax_utt = fig.add_subplot(gs[0, 4:6])
    ax_A = fig.add_subplot(gs[1, 1:5])

    axes = [ax_dxx, ax_dyy, ax_utt, ax_A]

    ax_dxx.spy(Dxx, markersize=10, color="royalblue")
    ax_dxx.set_title(rf"$D_{{xx}}$, {Dxx.shape[0]} $\times$ {Dxx.shape[1]}")

    ax_dyy.spy(Dyy, markersize=10, color="royalblue")
    ax_dyy.set_title(rf"$D_{{yy}}$, {Dyy.shape[0]} $\times$ {Dyy.shape[1]}")

    ax_utt.spy(Dtt, markersize=5, color="royalblue")  # type: ignore
    ax_utt.set_title(rf"$D_{{tt}}$, {Dtt.shape[0]} $\times$ {Dtt.shape[1]}")

    ax_A.spy(A, markersize=0.5, color="royalblue")
    ax_A.set_title(rf"$A$, {A.shape[0]} $\times$ {A.shape[1]}")  # type: ignore
    annotate_bttb(ax_A, ns=ns)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # draw little arrow
    add_ij_axes(axes[0])

    fig.suptitle("Sparse structure of the wave equation operators", fontsize=20)
    fig.tight_layout()
    fig.savefig(OUTFILE, dpi=200)
    print(f"Saved figure to {OUTFILE}")


if __name__ == "__main__":
    main()

"""
Figure 2: a plot demonstrating the singular value distribution and structure
of the global operator before and after preconditioning
"""

from odil_wave import Grid, HomogeneousModel, WaveEquation, Wavefield
from odil_wave.optimisation.precond import AlphaCirculantPreconditioner

import scipy.linalg as sl
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy.sparse.linalg import LinearOperator

OUTFILE = "fig2_sv_distribution.png"

NX, NY = 3, 3

UNPRECOND_COLOR = "royalblue"
PRECOND_COLOR = "orange"


def make_problem(
    nx=8,
    ny=8,
    t_max=0.5,
    cfl_safety=0.8,
    c_min=1.0,
    c_max=10.0,
    time_order=2,
    space_order=2,
):
    grid = Grid(
        nx=nx, ny=ny, t_max=t_max, cfl_safety=cfl_safety, c_min=c_min, c_max=c_max
    )

    model = HomogeneousModel(grid, background_c=c_min)
    wf = Wavefield(grid)
    we = WaveEquation(wf, model, time_order=time_order, space_order=space_order)

    return grid, model, we


def make_left_preconditioned_operator(A, precond):
    n = A.shape[0]

    def matvec(x):
        return precond.matvec(A.matvec(x))

    return LinearOperator(
        shape=(n, n),
        matvec=matvec,  # type: ignore
        dtype=np.float64,
    )


def M_to_dense(op, n):
    ident = np.eye(n, dtype=op.dtype)
    return np.column_stack([op.matvec(ident[:, j]) for j in range(n)])


def make_forward_circulant_operator(precond):
    """Forward action of the alpha-circulant approximation M itself (not M^-1)."""
    B0, B1, B2 = precond.blocks
    n, ns, alpha = precond.n, precond.ns, precond.alpha
    gamma = alpha ** (1.0 / n)
    d = gamma ** np.arange(n)
    z = gamma * np.exp(-2j * np.pi * np.arange(n // 2 + 1) / n)
    symbols = [(B0 + zk * B1 + zk**2 * B2).toarray() for zk in z]

    def matvec(v):
        V = v.reshape(n, ns) * d[:, None]
        Vh = np.fft.rfft(V, axis=0)
        Wh = np.stack([symbols[k] @ Vh[k] for k in range(Vh.shape[0])])
        W = np.fft.irfft(Wh, n=n, axis=0) / d[:, None]
        return W.ravel()

    return LinearOperator(
        shape=(n * ns, n * ns), matvec=matvec, dtype=np.float64  # type: ignore
    )


def compute_cond(singular_vals):
    return singular_vals[0] / singular_vals[-1]


def plot_singular_vals(ax, s_A, s_A_precond):
    """Plot log(singular_vals) for the two matrices"""
    ax.semilogy(s_A, color=UNPRECOND_COLOR)
    ax.semilogy(s_A_precond, color=PRECOND_COLOR)
    ax.set_title(r"Singular value distribution", fontsize=16)
    ax.set_ylabel("Singular values", fontsize=14)
    ax.set_xlabel("Singular value index", fontsize=14)
    ax.set_yscale("log")


def spy(ax, matrix, color="royalblue", threshold=1e-8):
    """Plot a thresholded structure view of a dense or sparse matrix."""
    if sp.issparse(matrix):
        pattern = np.abs(matrix.toarray()) > threshold
    else:
        pattern = np.abs(np.real_if_close(np.asarray(matrix))) > threshold
    cmap = LinearSegmentedColormap.from_list("spy", ["white", color])
    ax.imshow(pattern.astype(float), cmap=cmap, origin="upper", interpolation="none")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def label_panel(ax, label):
    """Tag a panel with a bold (a)/(b)/... label for subfigure referencing."""
    ax.text(
        -0.05,
        1.05,
        f"({label})",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="right",
    )

def plot_eigval_scatter(ax, eig, which="unprecond"):
    """Plot a complex plane scatter plot of eigenvalues"""

    # unpack config options
    config = {"unprecond": {"title": r"Eigenvalues of $A$", "colour": UNPRECOND_COLOR},
              "precond": {"title": r"Eigenvalues of $M^{-1}A$", "colour": PRECOND_COLOR}}[which]
    
    # unit circle for ref
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="gray", linestyle="--", linewidth=1)

    # plot
    ax.scatter(eig.real, eig.imag, s=18, color=config["colour"], alpha=0.8, edgecolors="none")

    # inset for tight clustering
    if which == "precond":
        axins = ax.inset_axes([0.55, 0.4, 0.2, 0.2])
        axins.plot(np.cos(theta), np.sin(theta), color="gray", linestyle="--", linewidth=0.8, zorder=5)
        axins.scatter(eig.real, eig.imag, s=20, color=config["colour"], alpha=0.8, edgecolors="none")

        pad = 0.025
        imag_center = np.median(eig.imag)
        axins.set_xlim(1 - pad, 1 + pad)
        axins.set_ylim(imag_center - pad, imag_center + pad)
        axins.set_aspect("equal")
        axins.set_xticks([])
        axins.set_yticks([])

        axins2 = ax.inset_axes([0.15, 0.3, 0.3, 0.4])
        axins2.plot(np.cos(theta), np.sin(theta), color="gray", linestyle="--", linewidth=0.8)
        axins2.scatter(eig.real, eig.imag, s=15, color=config["colour"], alpha=0.8, edgecolors="none")

        axins2.set_xticks([])
        axins2.set_yticks([])

        mult = 0.01
        axins2.set_xlim(1 - pad * mult, 1 + pad * mult)
        axins2.set_ylim(imag_center - pad * mult, imag_center + pad * mult)
        axins2.set_aspect("equal")
        

        ax.indicate_inset_zoom(axins, edgecolor="black")
        axins.indicate_inset_zoom(axins2, edgecolor="black")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_aspect("equal")
    ax.set_title(config["title"], fontsize=16)
    ax.set_xlabel("Re", fontsize=14)
    ax.set_ylabel("Im", fontsize=14)
        

def plot_sval_dist(ax, s_A, s_A_precond):
    bins = np.logspace(
        np.log10(min(s_A.min(), s_A_precond.min())),
        np.log10(max(s_A.max(), s_A_precond.max())),
        26,
    )
    ax.hist(s_A, bins=bins, alpha=0.6, color=UNPRECOND_COLOR)
    ax.hist(s_A_precond, bins=bins, alpha=0.6, color=PRECOND_COLOR)
    ax.set_title("Histogram of singular values", fontsize=16)
    ax.set_xlabel("Singular values", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_xscale("log")


def main():
    _, _, wave_eq = make_problem(nx=NX, ny=NY)

    # build the reduced operator and preconditioner on the same space
    A = wave_eq.reduced_operator()
    precond = AlphaCirculantPreconditioner.from_wave_equation(wave_eq, alpha=1e-4)

    A_dense = M_to_dense(A, A.shape[0])

    # M^-1 A is the preconditioned operator for the conditioning plots
    M_op = make_left_preconditioned_operator(A, precond)
    MA_dense = M_to_dense(M_op, A.shape[0])

    # M itself: the sparse alpha-circulant approximation of A, used for the
    # structure comparison
    M_fwd_op = make_forward_circulant_operator(precond)
    M_dense = M_to_dense(M_fwd_op, A.shape[0])

    s_A = sl.svdvals(A_dense)
    cond_A = compute_cond(s_A)

    s_MA = sl.svdvals(MA_dense)
    cond_MA = compute_cond(s_MA)

    eig_A = sl.eigvals(A_dense)
    eig_MA = sl.eigvals(MA_dense)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    spy(axes[0, 0], A_dense, color=UNPRECOND_COLOR)
    axes[0, 0].set_title(
        rf"$A$, {A_dense.shape[0]} $\times$ {A_dense.shape[1]}, "
        + rf"$\quad \kappa(A) = {cond_A:.1e}$",
        fontsize=16,
    )
    label_panel(axes[0, 0], "a")

    spy(axes[0, 1], M_dense, color=PRECOND_COLOR)
    axes[0, 1].set_title(
        rf"$M$, {M_dense.shape[0]} $\times$ {M_dense.shape[1]}, "
        + rf"$\quad \kappa(M^{{-1}}A) = {cond_MA:.1e}$",
        fontsize=16,
    )
    label_panel(axes[0, 1], "b")

    #plot_singular_vals(axes[1, 0], s_A, s_MA)
    plot_eigval_scatter(axes[1, 0], eig_A, "unprecond")
    label_panel(axes[1, 0], "c")

    #plot_sval_dist(axes[1, 1], s_A, s_MA)
    plot_eigval_scatter(axes[1, 1], eig_MA, "precond")
    label_panel(axes[1, 1], "d")

    fig.tight_layout()

    # push the top row spy plots out to the outer edges of the bottom row
    pos_tl, pos_tr = axes[0, 0].get_position(), axes[0, 1].get_position()
    pos_bl, pos_br = axes[1, 0].get_position(), axes[1, 1].get_position()

    axes[0, 0].set_position([pos_bl.x0, pos_tl.y0, pos_tl.width, pos_tl.height])
    axes[0, 1].set_position(
        [pos_br.x1 - pos_tr.width, pos_tr.y0, pos_tr.width, pos_tr.height]
    )

    gap_x = (pos_bl.x0 + pos_tl.width + pos_br.x1 - pos_tr.width) / 2
    gap_y = pos_tl.y0 + pos_tl.height / 2

    legend_handles = [
        Patch(facecolor=UNPRECOND_COLOR, label=r"Unpreconditioned"),
        Patch(facecolor=PRECOND_COLOR, label=r"Preconditioned"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(gap_x, gap_y),
        frameon=False,
        fontsize=14,
    )

    fig.savefig(OUTFILE, dpi=1200, bbox_inches="tight")
    print(f"Saved figure to {OUTFILE}")


if __name__ == "__main__":
    main()

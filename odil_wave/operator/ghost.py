from dataclasses import dataclass, field

import scipy.sparse as sp
import numpy as np


def _lagrange_weights(p: float, nodes: np.ndarray) -> np.ndarray:
    """Lagrange polynomial weights at point p using given node positions.

    This is required for grid extrapolation. We use ghost nodes to avoid stencil
    truncation at boundaries, and extrapolate behaviour in the domain to outside
    the domain (on the ghost nodes) using Lagrange extrapolation.
    """
    m = len(nodes)
    weights = np.ones(m)

    # compute lagrange polynomial weights
    # each iteration computes L_j(p) for one node
    for j in range(m):
        mask = np.arange(m) != j  # for all indices except j

        num = np.prod(p - nodes[mask])  # \prod (p - x_k)
        den = np.prod(nodes[j] - nodes[mask])  # \prod (x_j - x_k)

        weights[j] = num / den  # w_j = \prod (p - x_k) / \prod (x_j - x_k)

    return weights


@dataclass
class GhostFill:
    """Extend an array of `n` values by `ghost_width` extrapolated values per side.

    Ghost values are computed by Lagrange extrapolation from the nearest
    interior nodes, avoiding stencil truncation at domain boundaries.

    Parameters
    ----------
    n : int
        Number of physical (interior) nodes along this direction.
    ghost_width : int
        Number of ghost nodes appended at each end.

    Attributes
    ----------
    G : scipy.sparse.csr_array
        (n + 2*ghost_width, n) ghost extension matrix.
    """

    n: int  # physical nodes in this direction
    ghost_width: int  # g nodes per side

    G: sp.csr_array = field(init=False)  # (n + 2g, n)

    def __post_init__(self):
        self.G = self._build()

    def _build(self) -> sp.csr_array:
        """Build the ghost node array G (n + 2g, n)"""
        n, g = self.n, self.ghost_width

        if g == 0:
            return sp.eye(n, format="csr").tocsr()

        # k is extrapolation order, must be >=3 so the second normal derivative
        # survives. cubic cap avoids high order instability
        k = min(4, max(2 * g, 3), n)

        # g ghost nodes left and right of the domain
        left_nodes = np.arange(k, dtype=float)
        right_nodes = np.arange(n - k, n, dtype=float)

        # left ghost block: g rows, weights in first 2g columns
        left = np.zeros((g, n))
        for i in range(g):
            # compute extrapolation weights for left nodes
            left[i, :k] = _lagrange_weights(float(i - g), left_nodes)

        # right ghost block: g rows, weights in last 2g columns
        right = np.zeros((g, n))
        for i in range(g):
            # compute extrapolation weights for right nodes
            right[i, n - k :] = _lagrange_weights(float(n + i), right_nodes)

        # stack row-wise such that left and right weights pad identity (unextrapolated)
        return sp.vstack([left, sp.eye(n), right], format="csr")  # type: ignore

    def expand_x(self, ny: int) -> sp.csr_array:
        """Extend the x direction ghost operator over a full 2D grid.

        Parameters
        ----------
        ny : int
            Number of nodes along y.

        Returns
        -------
        scipy.sparse.csr_array
            (ny*(n+2g), n*ny) ghost extension operator for the x direction.
        """
        return sp.kron(self.G, sp.eye(ny)).tocsr()

    def expand_y(self, nx: int) -> sp.csr_array:
        """Extend the y direction ghost operator over a full 2D grid.

        Parameters
        ----------
        nx : int
            Number of nodes along x.

        Returns
        -------
        scipy.sparse.csr_array
            (nx*(n+2g), nx*n) ghost extension operator for the y direction.
        """
        return sp.kron(sp.eye(nx), self.G).tocsr()

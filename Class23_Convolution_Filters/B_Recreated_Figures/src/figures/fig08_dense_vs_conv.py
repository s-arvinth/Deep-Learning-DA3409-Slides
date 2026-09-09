"""A convolution is a fully connected layer with most of it deleted.

The layout of Prince (2023), Fig. 10.4, drawn for this deck: six inputs
x_1..x_6, and above each weight matrix the bipartite graph it encodes,
with every edge in the colour of the matrix entry it corresponds to.

(a-b) A fully connected layer: 36 free weights, 36 colours.
(c-d) A convolution with a kernel of three: the matrix is banded, and
      the band carries the same three colours down its whole length,
      because the weights are TIED and not merely sparse.  Three free
      numbers describe the whole matrix.
(e-f) The same kernel with stride two: every other row of (d) is
      removed, so there are three outputs and still three numbers.

The matrices are produced by `conv_matrix`, the same function the deck
checks against `conv1d`; nothing is drawn by hand.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from common import style as S
from common.models import conv_matrix

NAME = "dense_vs_conv"

N = 6
KCOL = [S.L1, S.OUTC, S.L2]          # the three kernel weights
DENSE_CMAP = plt.get_cmap("viridis")
RN = 0.40                            # node radius
DY = 0.95                            # row spacing in the graphs
XR = 3.6                             # x of the output column
FS = 13


def _node(ax, xy, text):
    ax.add_patch(Circle(xy, RN, fc="white", ec="black", lw=1.1, zorder=5))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=FS,
            zorder=6)


def _graph(ax, colours, n_out, title):
    """colours[i][j] is the edge colour from x_j to z_i, or None.
    Both columns are centred on y = 0."""
    ys_in = DY * (np.arange(N)[::-1] - (N - 1) / 2)
    if n_out == N:
        ys_out = ys_in
    else:   # stride two: each output sits level with the centre it reads
        ys_out = np.array([ys_in[min(2 * i, N - 1)] for i in range(n_out)])
    for i in range(n_out):
        for j in range(N):
            c = colours[i][j]
            if c is None:
                continue
            ax.plot([RN, XR - RN], [ys_in[j], ys_out[i]], color=c, lw=1.5,
                    alpha=0.95, zorder=3)
    for j in range(N):
        _node(ax, (0, ys_in[j]), r"$x_{%d}$" % (j + 1))
    for i in range(n_out):
        _node(ax, (XR, ys_out[i]), r"$z_{%d}$" % (i + 1))
    half = DY * (N - 1) / 2 + RN + 0.25
    ax.set_xlim(-RN - 0.3, XR + RN + 0.3); ax.set_ylim(-half, half)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=14, pad=6)


def _matrix(ax, colours, n_out, title, key):
    """The matrix, centred on (0, 0); `key` is drawn underneath."""
    x0, y0 = -N / 2, n_out / 2
    for i in range(n_out):
        for j in range(N):
            c = colours[i][j]
            ax.add_patch(Rectangle((x0 + j, y0 - 1 - i), 1, 1,
                                   fc=c if c is not None else "white",
                                   ec="black", lw=0.9))
    for j in range(N):
        ax.text(x0 + j + 0.5, y0 + 0.22, r"$x_{%d}$" % (j + 1), ha="center",
                va="bottom", fontsize=FS)
    for i in range(n_out):
        ax.text(x0 - 0.22, y0 - 0.5 - i, r"$z_{%d}$" % (i + 1), ha="right",
                va="center", fontsize=FS)
    # the key, centred below the matrix
    yk = -N / 2 - 1.0
    if key == "kernel":
        xs = np.array([-2.4, -0.4, 1.6])
        for x, c, k in zip(xs, KCOL, range(3)):
            ax.add_patch(Rectangle((x, yk - 0.25), 0.5, 0.5, fc=c, ec="black",
                                   lw=0.8))
            ax.text(x + 0.65, yk, r"$w_{%d}$" % (k + 1), va="center",
                    fontsize=FS)
    else:
        grad = np.linspace(0, 1, 64)
        for g in grad:
            ax.add_patch(Rectangle((-2.6 + 3.2 * g, yk - 0.25), 3.2 / 64 + 0.01,
                                   0.5, fc=DENSE_CMAP(g), ec="none"))
        ax.add_patch(Rectangle((-2.6, yk - 0.25), 3.2, 0.5, fc="none",
                               ec="black", lw=0.8))
        ax.text(0.8, yk, "36 different values", va="center", fontsize=FS - 1)
    half = N / 2 + 1.0
    ax.set_xlim(-half - 0.3, half + 0.3); ax.set_ylim(yk - 0.55, half + 0.15)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=14, pad=6)


def build():
    S.use()
    rng = np.random.default_rng(23)
    dense = rng.uniform(size=(N, N))
    dense_cols = [[DENSE_CMAP(dense[i, j]) for j in range(N)] for i in range(N)]

    w = np.array([1.0, 2.0, 3.0])            # labels, not values
    M1 = conv_matrix(N, w, s=1, p=1)
    M2 = conv_matrix(N, w, s=2, p=1)
    def cols(M):
        return [[KCOL[int(M[i, j]) - 1] if M[i, j] != 0 else None
                 for j in range(N)] for i in range(M.shape[0])]
    print("dense: %d free weights; kernel 3: %d; kernel 3, stride 2: %d "
          "(outputs %d, %d, %d)" % (N * N, 3, 3, N, M1.shape[0], M2.shape[0]))

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.9))
    _graph(axes[0, 0], dense_cols, N, "(a) fully connected")
    _matrix(axes[1, 0], dense_cols, N, "(b) its weight matrix: 36 free", "dense")
    _graph(axes[0, 1], cols(M1), M1.shape[0], "(c) kernel of three")
    _matrix(axes[1, 1], cols(M1), M1.shape[0], "(d) banded and tied: 3 free",
            "kernel")
    _graph(axes[0, 2], cols(M2), M2.shape[0], "(e) kernel of three, stride two")
    _matrix(axes[1, 2], cols(M2), M2.shape[0], "(f) every other row: 3 free",
            "kernel")
    fig.tight_layout(w_pad=0.6, h_pad=0.3)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

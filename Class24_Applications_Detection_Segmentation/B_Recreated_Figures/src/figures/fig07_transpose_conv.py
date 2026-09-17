"""Transpose convolution, drawn.

The layout of Bishop & Bishop (2024), Fig. 10.30, redrawn: a 2 x 2
input, a 3 x 3 kernel and an output stride of two give a 5 x 5 output.
The red output patch is the kernel times the red input unit, the blue
patch likewise, and where the two overlap (purple) their contributions
are summed.  The build also checks, numerically, that this scattering
equals M^T z for the matrix M of the 5 x 5 -> 2 x 2 strided
convolution (Bishop's Exercise 10.13); the picture shows only the
geometry.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common import bishopdraw as D
from common.models import conv2d_matrix, transpose_conv

NAME = "transpose_conv"


def build():
    S.use()
    W = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float)
    Z = np.array([[1.0, 2.0], [3.0, 4.0]])
    M, _ = conv2d_matrix(5, W, 2)
    assert np.allclose(transpose_conv(Z, W, 2, 5).ravel(), M.T @ Z.ravel())
    print("scattered sum equals M^T z: checked")

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    # input 2 x 2: red (0,0), blue (0,1), yellow rest
    xi, ti = 0.0, 1.0
    D.draw_grid(ax, xi, ti, 2, {(0, 0): D.RED, (0, 1): D.BLUE})
    r_in = D.block_outline(ax, xi, ti, 0, 0, 1)
    b_in = D.block_outline(ax, xi, ti, 0, 1, 1)
    # output 5 x 5: red rows 0-2 cols 0-2, blue rows 0-2 cols 2-4, purple col 2
    xo, to = 3.6, 1.6
    f = {}
    for r in range(3):
        for c in range(3):
            f[(r, c)] = D.RED
        for c in range(2, 5):
            f[(r, c)] = D.PURPLE if c == 2 else D.BLUE
    D.draw_grid(ax, xo, to, 5, f)
    r_out = D.block_outline(ax, xo, to, 0, 0, 3)
    b_out = D.block_outline(ax, xo, to, 0, 2, 3)
    D.connect(ax, r_in, r_out, color="#D02020", lw=0.8)
    D.connect(ax, b_in, b_out, color="#2020D0", lw=0.8)
    ax.text(xi - 0.1, ti + 2 * D.B + 0.45, "input $2 \\times 2$", fontsize=13, ha="left")
    ax.text(xo + 0.2, to + 5 * D.B + 0.45, "output $5 \\times 5$", fontsize=13, ha="left")
    ax.annotate("", xy=(3.2, -1.4), xytext=(0.6, -1.4),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4))
    ax.set_xlim(-0.3, 6.4); ax.set_ylim(-2.4, 3.6); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

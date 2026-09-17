"""How the effective receptive field grows with depth.

The layout of Bishop & Bishop (2024), Fig. 10.9, redrawn in the deck's
colours: three layers of units, each unit reading three of the layer
below (a kernel of three, stride one).  The maroon unit at the top of
the output layer depends on three units of the middle layer and, through
them, on five of the input layer -- the units and edges in maroon.  The
counts are computed by `stack_rf`, not typed in.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from common import style as S
from common.models import stack_rf

NAME = "receptive_field"
N0, N1, N2 = 7, 4, 2                  # units per layer (top-aligned)
K = 3
DY, DX, R = 1.0, 2.2, 0.30
INFIELD_FC, INFIELD_EC = "#F3C9C9", S.OUTC
OUT_FC, OUT_EC = "#CFE0EA", S.L1


def build():
    S.use()
    rf1, rf2 = stack_rf(K, 1), stack_rf(K, 2)
    assert (rf1, rf2) == (3, 5)
    print("receptive field: %d in the middle layer, %d in the input" % (rf1, rf2))
    layers = [N0, N1, N2]
    ys = [np.array([-i * DY for i in range(n)]) for n in layers]
    xs = [0.0, DX, 2 * DX]
    # which units feed the top output unit
    field = [set(range(rf2)), set(range(rf1)), {0}]

    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    for l in range(2):
        for j in range(layers[l + 1]):
            for t in range(K):
                i = j + t
                hot = (j in field[l + 1]) and (i in field[l])
                ax.annotate("", xy=(xs[l + 1] - R, ys[l + 1][j]),
                            xytext=(xs[l] + R, ys[l][i]),
                            arrowprops=dict(arrowstyle="-|>", lw=2.0 if hot else 0.8,
                                            color=S.OUTC if hot else "black",
                                            shrinkA=0, shrinkB=0), zorder=1)
    for l in range(3):
        for i in range(layers[l]):
            hot = i in field[l]
            ax.add_patch(Circle((xs[l], ys[l][i]), R, fc=INFIELD_FC if hot else OUT_FC,
                                ec=INFIELD_EC if hot else OUT_EC, lw=1.6, zorder=3))
    # brace on the input field
    y0, y1 = ys[0][rf2 - 1] - 0.45, ys[0][0] + 0.45
    ax.plot([-0.75, -0.95, -0.95, -0.75], [y1, y1, y0, y0], color="black", lw=1.0)
    ax.plot([-0.95, -1.15], [(y0 + y1) / 2] * 2, color="black", lw=1.0)
    ax.text(-1.3, (y0 + y1) / 2, "receptive field", rotation=90, ha="center",
            va="center", fontsize=12)
    for l, lab in enumerate(("input", "hidden", "output")):
        ax.text(xs[l], 0.9, lab, ha="center", fontsize=12, color=S.GREY)
    ax.set_xlim(-1.8, 2 * DX + 0.6); ax.set_ylim(-(N0 - 1) * DY - 0.6, 1.3)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

"""regions — the separating line 0.9x1+2x2=1 with its two region inequalities and normal w

Used in: Class 3 (B), slide(s): "Weights tilt the separating line"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python regions.py            -> output/regions.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

from common import L1, L2, OUTC, save

NAME = "regions"


def build():
    w0, w1, w2 = -1.0, 0.9, 2.0          # 0.9 x1 + 2 x2 >= 1
    g = np.linspace(-0.6, 2.2, 400)
    X, Y = np.meshgrid(g, g)
    side = w0 + w1 * X + w2 * Y

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    cf = ax.contourf(X, Y, (side >= 0).astype(float), levels=[-.5, .5, 1.5],
                     colors=["#EFECF3", "#F3E2E2"])
    cf.set_rasterized(True)
    ax.contour(X, Y, side, levels=[0], colors=[L2], linewidths=2.0)

    # region labels, each carrying its inequality
    ax.text(0.12, 0.12, "unit stays silent\n$0.9x_1+2x_2 < 1$",
            color=L1, fontsize=10.5, ha="left", va="bottom")
    ax.text(1.98, 1.55, "unit fires\n$0.9x_1+2x_2 \\geq 1$",
            color=OUTC, fontsize=10.5, ha="right", va="top")
    # the boundary equation, written along the line
    ax.text(1.62, 0.02, "$0.9x_1+2x_2 = 1$", color=L2, fontsize=10.5,
            rotation=-24, ha="center", va="bottom")
    # weight vector, normal to the boundary
    n = np.array([w1, w2]); n = n / np.linalg.norm(n)
    foot = np.array([0.55, 0.25])
    ax.arrow(*foot, *(n * 0.55), color=L2, lw=1.6, head_width=0.08,
             length_includes_head=True, zorder=5)
    ax.annotate(r"$w=(0.9,\,2)$", foot + n * 0.55,
                textcoords="offset points", xytext=(6, 4), color=L2,
                fontsize=10)
    ax.set(xlim=(-0.6, 2.2), ylim=(-0.6, 2.2), xlabel=r"$x_1$",
           ylabel=r"$x_2$")
    ax.set_aspect("equal", "box")
    return save(fig, NAME)


if __name__ == "__main__":
    build()

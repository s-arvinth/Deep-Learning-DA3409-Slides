"""duality — input space (w fixed) and weight space (x fixed) side by side

Used in: Class 3 (B), slide(s): "Two dual viewpoints"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python duality.py            -> output/duality.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

from common import L2, OUTC, GREY, save

NAME = "duality"


def build():
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))

    # (a) input space: w is fixed, the inputs x vary
    w = np.array([0.0, 1.0, 1.4])
    g = np.linspace(-2, 2, 300)
    X, Y = np.meshgrid(g, g)
    side = w[0] + w[1] * X + w[2] * Y
    cf = ax[0].contourf(X, Y, (side >= 0).astype(float),
                        levels=[-.5, .5, 1.5],
                        colors=["#EFECF3", "#F3E2E2"])
    cf.set_rasterized(True)
    ax[0].contour(X, Y, side, levels=[0], colors=[L2], linewidths=1.9)
    n = w[1:] / np.linalg.norm(w[1:])
    ax[0].arrow(0, 0, *(n * 0.9), color=L2, lw=1.6, head_width=0.1,
                length_includes_head=True, zorder=5)
    ax[0].annotate(r"$w$", n * 0.9, textcoords="offset points",
                   xytext=(6, 4), color=L2, fontsize=11)
    ax[0].scatter([1.1], [0.9], s=95, marker="^", facecolor=OUTC,
                  edgecolor="white", lw=0.9, zorder=6)
    ax[0].annotate(r"$x$", (1.1, 0.9), textcoords="offset points",
                   xytext=(7, -12), color=OUTC, fontsize=11)
    ax[0].set(xlim=(-2, 2), ylim=(-2, 2), xlabel=r"$x_1$", ylabel=r"$x_2$",
              title=r"(a)  input space: $w$ fixed")

    # (b) weight space: x is fixed, the weights w vary
    x = np.array([0.0, 1.1, 0.9])
    W1, W2 = np.meshgrid(g, g)
    side2 = x[1] * W1 + x[2] * W2
    cf2 = ax[1].contourf(W1, W2, (side2 >= 0).astype(float),
                         levels=[-.5, .5, 1.5],
                         colors=["#EFECF3", "#F3E2E2"])
    cf2.set_rasterized(True)
    ax[1].contour(W1, W2, side2, levels=[0], colors=[OUTC], linewidths=1.9)
    nx = x[1:] / np.linalg.norm(x[1:])
    ax[1].arrow(0, 0, *(nx * 0.9), color=OUTC, lw=1.6, head_width=0.1,
                length_includes_head=True, zorder=5)
    ax[1].annotate(r"$x$", nx * 0.9, textcoords="offset points",
                   xytext=(6, 4), color=OUTC, fontsize=11)
    ax[1].scatter([0.7], [1.0], s=85, marker="o", facecolor="white",
                  edgecolor=L2, lw=1.8, zorder=6)
    ax[1].annotate(r"$w$", (0.7, 1.0), textcoords="offset points",
                   xytext=(7, -12), color=L2, fontsize=11)
    ax[1].set(xlim=(-2, 2), ylim=(-2, 2), xlabel=r"$w_1$", ylabel=r"$w_2$",
              title=r"(b)  weight space: $x$ fixed")
    for a in ax:
        a.set_aspect("equal", "box")
        a.axhline(0, color=GREY, lw=0.4, ls=":")
        a.axvline(0, color=GREY, lw=0.4, ls=":")
    fig.tight_layout(w_pad=6.0)

    # --- two short arrows in the gutter, tying the pictures together --
    from matplotlib.patches import FancyArrowPatch
    for y, col, lab in [(0.66, L2,   r"$w$: vector $\to$ point"),
                        (0.34, OUTC, r"$x$: point $\to$ normal")]:
        fig.add_artist(FancyArrowPatch(
            (0.455, y), (0.545, y), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=13, lw=1.5, color=col))
        fig.text(0.50, y + 0.055, lab, color=col, fontsize=9,
                 ha="center")
    return save(fig, NAME)


if __name__ == "__main__":
    build()

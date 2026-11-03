"""update_ws — one update on the AND error map and as a constraint in weight space

Used in: Class 3 (B), slide(s): "The update, seen in weight space"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python update_ws.py            -> output/update_ws.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

from common import L1, OUTC, ACC, AND_TH, and_error, save

NAME = "update_ws"


def build():
    g = np.linspace(-0.5, 1.5, 220)
    W1, W2 = np.meshgrid(g, g)
    E = and_error(W1, W2)
    w  = np.array([0.15, 0.45])          # misclassifies (1,1)
    x  = np.array([1.0, 1.0])            # the offending pattern
    w2 = w + 0.45 * x

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.5))

    # (a) the step drawn on the error map
    m = ax[0].pcolormesh(W1, W2, E, cmap="plasma", shading="auto", alpha=.7)
    m.set_rasterized(True)
    ax[0].contour(W1, W2, E, levels=[0.5], colors="white", linewidths=1.8)
    ax[0].annotate("", xy=w2, xytext=w,
                   arrowprops=dict(arrowstyle="-|>", color="white", lw=3.2))
    ax[0].annotate("", xy=w2, xytext=w,
                   arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.8))
    ax[0].scatter(*w,  s=80, marker="s", color=L1, edgecolor="white",
                  zorder=5, label=r"$w$ (2 errors)")
    ax[0].scatter(*w2, s=110, marker="*", color=OUTC, edgecolor="white",
                  zorder=5, label=r"$w'=w+\eta x$")
    ax[0].legend(loc="lower left", fontsize=8, framealpha=.9,
                 facecolor="white", edgecolor="none")
    ax[0].set(xlabel=r"$w_1$", ylabel=r"$w_2$",
              title="(a)  the step, on the error map")
    ax[0].set_aspect("equal", "box")

    # (b) the same step in weight space: x defines the constraint plane
    gg = np.linspace(-0.5, 1.5, 300)
    A, B = np.meshgrid(gg, gg)
    ok = (A * x[0] + B * x[1]) >= AND_TH
    cf = ax[1].contourf(A, B, ok.astype(float), levels=[-.5, .5, 1.5],
                        colors=["#EFECF3", "#F3E2E2"])
    cf.set_rasterized(True)
    ax[1].contour(A, B, A * x[0] + B * x[1], levels=[AND_TH],
                  colors=[OUTC], linewidths=1.9)
    nx = x / np.linalg.norm(x)
    ax[1].annotate("", xy=w2, xytext=w,
                   arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.8))
    ax[1].scatter(*w,  s=80, marker="s", color=L1, edgecolor="white",
                  zorder=5)
    ax[1].scatter(*w2, s=110, marker="*", color=OUTC, edgecolor="white",
                  zorder=5)
    ax[1].annotate(r"$w\cdot x \geq \theta$", (1.15, 1.2), color=OUTC,
                   fontsize=9.5, ha="center")
    ax[1].annotate(r"$w\cdot x < \theta$", (0.05, 0.02), color=L1,
                   fontsize=9.5, ha="left")
    ax[1].set(xlabel=r"$w_1$", ylabel=r"$w_2$",
              title=r"(b)  the constraint $x$ imposes")
    ax[1].set_aspect("equal", "box")
    fig.tight_layout(w_pad=2.0)
    return save(fig, NAME)


if __name__ == "__main__":
    build()

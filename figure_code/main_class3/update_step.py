"""update_step — one PLA update w' = w + x seen in input space

Used in: Class 3 (B), slide(s): "Why the update works"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python update_step.py            -> output/update_step.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

from common import L2, OUTC, ACC, GREY, boundary, save

NAME = "update_step"


def build():
    x = np.array([1.0, 1.6, 0.9])          # extended, a misclassified P
    w = np.array([0.2, -1.1, 0.6])         # w.x < 0  -> wrong
    w2 = w + x
    xs = np.linspace(-2.4, 2.4, 20)

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.3))

    for a, ww, ttl in [(ax[0], w,  r"(a)  before: $w\cdot x < 0$"),
                       (ax[1], w2, r"(b)  after: $w' = w + x$")]:
        boundary(a, ww, xs, color=L2, lw=1.8)
        # shade the positive half-plane
        X, Y = np.meshgrid(np.linspace(-2.4, 2.4, 240),
                           np.linspace(-2.4, 2.4, 240))
        side = ww[0] + ww[1] * X + ww[2] * Y
        a.contourf(X, Y, (side > 0).astype(float), levels=[.5, 1.5],
                   colors=[OUTC], alpha=.09)
        a.contour(X, Y, side, levels=[0], colors=[L2], linewidths=0)
        a.arrow(0, 0, ww[1] * .7, ww[2] * .7, color=L2, lw=1.4,
                head_width=0.14, length_includes_head=True)
        a.annotate(r"$w$", (ww[1] * .7, ww[2] * .7),
                   textcoords="offset points", xytext=(6, 4),
                   color=L2, fontsize=10)
        a.scatter([x[1]], [x[2]], s=95, marker="^", facecolor=OUTC,
                  edgecolor="white", lw=0.9, zorder=5)
        a.annotate(r"$x \in P$", (x[1], x[2]), textcoords="offset points",
                   xytext=(8, -12), color=OUTC, fontsize=10)
        a.set(xlim=(-2.4, 2.4), ylim=(-2.4, 2.4), xlabel=r"$x_1$",
              ylabel=r"$x_2$", title=ttl)
        a.set_aspect("equal", "box")
        a.axhline(0, color=GREY, lw=0.4, ls=":")
        a.axvline(0, color=GREY, lw=0.4, ls=":")
    ax[1].annotate("", xy=(w2[1] * .7, w2[2] * .7), xytext=(w[1] * .7, w[2] * .7),
                   arrowprops=dict(arrowstyle="->", color=ACC, lw=1.4))
    for a in ax:
        a.text(0, -2.22, "shaded = where the unit fires", fontsize=8.5,
               color=GREY, ha="center")
    fig.tight_layout(w_pad=2.0)
    return save(fig, NAME)


if __name__ == "__main__":
    build()

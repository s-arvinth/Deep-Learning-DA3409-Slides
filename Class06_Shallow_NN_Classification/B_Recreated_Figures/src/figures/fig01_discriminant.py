#!/usr/bin/env python3
"""Geometry of a linear discriminant: normal direction and signed distance."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.data import two_gaussians
from common.models import fit_logistic

NAME = "discriminant"


def build():
    style.use()
    X, y, _ = two_gaussians(N=80, sep=4.0,
                            cov=((0.85, 0.30), (0.30, 0.70)))
    w = fit_logistic(X, y)
    w0, wv = w[0], w[1:]
    nrm = np.linalg.norm(wv)

    g = np.linspace(-5.5, 5.5, 400)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=34, marker="o",
               facecolor="white", edgecolor=L1, lw=1.3,
               label=r"class $\mathcal{C}_1$")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=34, marker="s",
               facecolor="white", edgecolor=OUTC, lw=1.3,
               label=r"class $\mathcal{C}_2$")
    ax.plot(g, -(w0 + wv[0] * g) / wv[1], color="black", lw=2.0,
            label=r"$a(\mathbf{x})=0$")

    # foot of the perpendicular from a chosen point
    p = np.array([-2.9, 2.6])
    r = (wv @ p + w0) / nrm
    foot = p - r * wv / nrm
    ax.plot([p[0], foot[0]], [p[1], foot[1]], color=ACC, lw=1.8, ls="--")
    ax.scatter(*p, s=70, color=ACC, zorder=5, edgecolor="white")
    ax.annotate(r"$r=\dfrac{a(\mathbf{x})}{\|\mathbf{w}\|}$",
                xy=(p[0] - 0.15, p[1] + 0.45), color=ACC, fontsize=18.2,
                ha="left")
    # the normal vector w, drawn from the boundary
    ytop = 2.6
    base = np.array([-(w0 + wv[1] * ytop) / wv[0], ytop])
    ax.annotate("", xy=base + 2.2 * wv / nrm, xytext=base,
                arrowprops=dict(arrowstyle="-|>", color=L2, lw=3.0,
                                mutation_scale=22))
    ax.annotate(r"$\mathbf{w}$",
                xy=base + 2.4 * wv / nrm + np.array([0.15, 0.0]),
                color=L2, fontsize=20.8, va="center")
    ax.legend(loc="lower right", fontsize=13.7, framealpha=.93,
              facecolor="white", edgecolor="none")
    ax.set(xlim=(-5.5, 5.5), ylim=(-4.5, 4.5),
           xlabel=r"$x_1$", ylabel=r"$x_2$")
    ax.set_title(r"$\mathbf{w}$ is normal to the boundary; "
                 r"$w_0$ sets its offset", fontsize=16.2)
    style.square(ax)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

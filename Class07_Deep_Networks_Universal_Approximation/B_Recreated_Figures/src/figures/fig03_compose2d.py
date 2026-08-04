#!/usr/bin/env python3
"""Composition in two dimensions: creasing an already-creased plane."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import net2d, f2, V1

NAME = "compose2d"


def build():
    style.use()
    g = np.linspace(-2, 2, 700)
    X1, X2 = np.meshgrid(g, g)
    a, y = net2d(X1, X2)
    yprime = f2(y.ravel()).reshape(y.shape)
    pattern = ((a[0] > 0) * 4 + (a[1] > 0) * 2 + (a[2] > 0)).astype(float)

    fig, ax = plt.subplots(1, 3, figsize=(11.6, 4.3))

    cf = ax[0].contourf(X1, X2, pattern, levels=np.arange(-.5, 8.5, 1),
                        cmap=style.REGION_CMAP)
    cf.set_rasterized(True)
    for k in range(3):
        w1, w2, b = V1[k]
        if abs(w2) > 1e-9:
            ax[0].plot(g, -(w1 * g + b) / w2, color=BRANCH[k], lw=2.0,
                       label=rf"$a_{k+1}=0$")
    ax[0].legend(loc="lower left", fontsize=11, framealpha=.92,
                 facecolor="white", edgecolor="none")
    ax[0].set(xlim=(-2, 2), ylim=(-2, 2), xlabel=r"$x_1$", ylabel=r"$x_2$")
    ax[0].set_title(r"(a)  layer 1: convex regions", fontsize=12.5)

    c = ax[1].contourf(X1, X2, y, levels=24, cmap=style.SEQ)
    c.set_rasterized(True)
    ax[1].contour(X1, X2, pattern, levels=np.arange(-.5, 8.5, 1),
                  colors="w", linewidths=0.8, alpha=.85)
    cb = fig.colorbar(c, ax=ax[1], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=12)
    ax[1].set(xlabel=r"$x_1$", ylabel=r"$x_2$")
    ax[1].set_title(r"(b)  its output: planar per region", fontsize=12.5)

    c2 = ax[2].contourf(X1, X2, yprime, levels=24, cmap=style.CAT)
    c2.set_rasterized(True)
    ax[2].contour(X1, X2, pattern, levels=np.arange(-.5, 8.5, 1),
                  colors="w", linewidths=0.8, alpha=.85)
    cb2 = fig.colorbar(c2, ax=ax[2], fraction=0.046, pad=0.03)
    cb2.ax.tick_params(labelsize=12)
    ax[2].set(xlabel=r"$x_1$", ylabel=r"$x_2$")
    ax[2].set_title(r"(c)  layer 2 subdivides each region", fontsize=12.5)

    for a_ in ax:
        a_.set_aspect("equal", "box")
    fig.tight_layout(w_pad=1.8)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

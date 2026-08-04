#!/usr/bin/env python3
"""The two-input network built up: hinges -> convex regions -> surface."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import shallow2d, V1
from matplotlib.colors import ListedColormap

NAME = "buildup2d"


def build():
    style.use()
    g = np.linspace(-2, 2, 600)
    X1, X2 = np.meshgrid(g, g)
    a, y = shallow2d(X1, X2)
    pattern = ((a[0] > 0) * 4 + (a[1] > 0) * 2 + (a[2] > 0)).astype(float)

    fig = plt.figure(figsize=(11.4, 4.3))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3, projection="3d")

    cm = ListedColormap(["#EFECF3", "#E1EAEF", "#F3E7E7", "#F6F1E5",
                         "#E9EDE9", "#EDE6F0", "#E4EEF1", "#F8F4EC"])
    cf = ax0.contourf(X1, X2, pattern, levels=np.arange(-.5, 8.5, 1), cmap=cm)
    cf.set_rasterized(True)
    for j in range(3):
        w1, w2, b = V1[j]
        if abs(w2) > 1e-9:
            ax0.plot(g, -(w1 * g + b) / w2, color=BRANCH[j], lw=2.0,
                     label=rf"$a_{j+1}=0$")
    ax0.legend(loc="lower left", fontsize=12.3, framealpha=.9,
               facecolor="white", edgecolor="none")
    ax0.set(xlim=(-2, 2), ylim=(-2, 2), xlabel=r"$x_1$", ylabel=r"$x_2$")
    ax0.set_title(r"(a)  $M=3$ hinges cut the plane into convex regions",
                  fontsize=14.3)

    m = ax1.pcolormesh(X1, X2, y, cmap=style.SEQ, shading="auto")
    m.set_rasterized(True)
    ax1.contour(X1, X2, pattern, levels=np.arange(-.5, 8.5, 1),
                colors="white", linewidths=0.8, alpha=.9)
    cb = fig.colorbar(m, ax=ax1, fraction=0.046, pad=0.03)
    cb.set_label(r"$\hat y$", fontsize=14.3)
    cb.ax.tick_params(labelsize=12.3)
    ax1.set(xlabel=r"$x_1$", ylabel=r"$x_2$")
    ax1.set_title(r"(b)  $\hat y$ is a single plane on each region",
                  fontsize=14.3)

    s = ax2.plot_surface(X1[::6, ::6], X2[::6, ::6], y[::6, ::6],
                         cmap=style.SEQ, linewidth=0, antialiased=True)
    s.set_rasterized(True)
    ax2.set_xlabel(r"$x_1$", fontsize=15.6)
    ax2.set_ylabel(r"$x_2$", fontsize=15.6)
    ax2.set_zlabel(r"$\hat y$", fontsize=15.6)
    ax2.set_title(r"(c)  the same surface, folded along the hinges",
                  fontsize=14.3)
    ax2.tick_params(labelsize=11.7)
    for pane in (ax2.xaxis, ax2.yaxis, ax2.zaxis):
        pane.pane.set_facecolor("white")
        pane.pane.set_edgecolor("#DDDDDD")
        pane._axinfo["grid"]["color"] = "#EEEEEE"
    ax2.view_init(elev=26, azim=-132)
    ax2.set_box_aspect((1, 1, 0.75), zoom=0.90)
    for axis in (ax0, ax1):
        axis.set_aspect("equal", "box")
    fig.tight_layout(w_pad=1.8)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

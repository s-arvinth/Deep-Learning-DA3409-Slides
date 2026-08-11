#!/usr/bin/env python3
"""The curse of dimensionality for fixed basis functions: a grid needs
S^D cells, so the number of parameters explodes with the input dimension."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

NAME = "curse_grid"
S = 4          # cells per axis


def build():
    style.use()
    fig = plt.figure(figsize=(15.0, 4.0))
    ax0 = fig.add_subplot(1, 4, 1)
    ax1 = fig.add_subplot(1, 4, 2)
    ax2 = fig.add_subplot(1, 4, 3, projection="3d")
    ax3 = fig.add_subplot(1, 4, 4)

    e = np.linspace(0, 1, S + 1)
    cmap = plt.get_cmap(style.CAT)

    # D = 1
    for i in range(S):
        ax0.add_patch(Rectangle((e[i], 0.35), 1 / S, 0.30,
                                facecolor=cmap(0.1 + 0.18 * i), alpha=.55,
                                edgecolor="white", lw=1.4))
    ax0.set(xlim=(0, 1), ylim=(0, 1), yticks=[], xlabel=r"$x_1$")
    for side in ("left", "right", "top"):
        ax0.spines[side].set_visible(False)
    ax0.set_title(rf"(a)  $D=1$:  ${S}$ cells", fontsize=12.5)

    # D = 2
    for i in range(S):
        for j in range(S):
            ax1.add_patch(Rectangle((e[i], e[j]), 1 / S, 1 / S,
                                    facecolor=cmap(0.05 + 0.055 * (i + j)),
                                    alpha=.55, edgecolor="white", lw=1.1))
    ax1.set(xlim=(0, 1), ylim=(0, 1), xlabel=r"$x_1$", ylabel=r"$x_2$")
    ax1.set_title(rf"(b)  $D=2$:  ${S**2}$ cells", fontsize=12.5)
    ax1.set_aspect("equal", "box")

    # D = 3 -- draw only the outer shell of cubes, else it is a solid block
    for i in range(S):
        for j in range(S):
            for k in range(S):
                if not (i in (0, S - 1) or j in (0, S - 1) or k in (0, S - 1)):
                    continue
                x0, y0, z0 = e[i], e[j], e[k]
                d = 1 / S
                pts = np.array([[x0, y0, z0], [x0 + d, y0, z0],
                                [x0 + d, y0 + d, z0], [x0, y0 + d, z0],
                                [x0, y0, z0 + d], [x0 + d, y0, z0 + d],
                                [x0 + d, y0 + d, z0 + d], [x0, y0 + d, z0 + d]])
                faces = [[pts[a] for a in f] for f in
                         ([0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                          [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4])]
                pc = Poly3DCollection(faces, alpha=.16,
                                      facecolor=cmap(0.05 + 0.05 * (i + j + k)),
                                      edgecolor="white", linewidths=0.5)
                ax2.add_collection3d(pc)
    ax2.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))
    ax2.set_xlabel(r"$x_1$", fontsize=12); ax2.set_ylabel(r"$x_2$", fontsize=12)
    ax2.set_zlabel(r"$x_3$", fontsize=12)
    style.tidy3d(ax2, ticks=(0.0, 0.5, 1.0), labelsize=11)
    ax2.view_init(elev=20, azim=-58)
    ax2.set_title(rf"(c)  $D=3$:  ${S**3}$ cells", fontsize=12.5)

    # the count
    D = np.arange(1, 21)
    for s, c, mk in zip([2, 4, 10], [L1, L2, OUTC], ["o", "s", "^"]):
        ax3.semilogy(D, float(s) ** D, color=c, lw=2.2, marker=mk, ms=6,
                     markevery=2, mec="white", mew=0.8,
                     label=rf"$S={s}$ per axis")
    ax3.axhline(1e9, color=GREY, lw=1.2, ls=":")
    ax3.annotate("a billion", xy=(1.5, 1.6e9), fontsize=11, color=GREY)
    ax3.legend(loc="lower right", fontsize=11)
    ax3.set(xlabel=r"input dimension $D$", ylabel=r"cells $S^{D}$")
    style.integer_ticks(ax3, "x", nbins=5)
    ax3.set_title(r"(d)  $S^{D}$ grows exponentially", fontsize=12.5)
    style.square(ax3)
    for a in (ax0, ax1):
        style.square(a)
    fig.tight_layout(w_pad=2.2)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

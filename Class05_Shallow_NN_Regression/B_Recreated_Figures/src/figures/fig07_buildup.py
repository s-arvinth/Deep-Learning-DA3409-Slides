#!/usr/bin/env python3
"""The scalar network computed stage by stage: a_j -> z_j -> weighted -> sum."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import shallow, joints, W1, W2

NAME = "buildup"


def build():
    style.use()
    x = np.linspace(-1, 1, 1400)
    a, z, yhat = shallow(x)

    fig, ax = plt.subplots(2, 3, figsize=(11.4, 7.4))
    for j in range(3):
        b, w = W1[j]
        ax[0, j].plot(x, a[j], color=GREY, lw=1.3, ls="--",
                      label=r"pre-activation $a_j$")
        ax[0, j].plot(x, z[j], color=BRANCH[j], lw=2.4,
                      label=r"activation $z_j=h(a_j)$")
        ax[0, j].axhline(0, color=GREY, lw=0.5, ls=":")
        ax[0, j].set(xlabel=r"$x$", ylabel=rf"$z_{j+1}$")
        ax[0, j].set_title(rf"({chr(97+j)})  $z_{j+1}=h({w:.0f}\,x{b:+.2f})$",
                           fontsize=16.2)
    ax[0, 0].legend(loc="upper left", fontsize=12.7)

    for j in range(3):
        ax[1, 0].plot(x, W2[j + 1] * z[j], color=BRANCH[j], lw=2.2,
                      label=rf"$w^{{(2)}}_{j+1}z_{j+1}$")
    ax[1, 0].axhline(0, color=GREY, lw=0.5, ls=":")
    ax[1, 0].legend(fontsize=13.0)
    ax[1, 0].set(xlabel=r"$x$", ylabel=r"$w^{(2)}_j z_j$")
    ax[1, 0].set_title(r"(d)  each unit scaled by its output weight",
                       fontsize=16.2)

    ax[1, 1].plot(x, yhat, color=OUTC, lw=2.6)
    ax[1, 1].axhline(0, color=GREY, lw=0.5, ls=":")
    ax[1, 1].set(xlabel=r"$x$", ylabel=r"$\hat y$")
    ax[1, 1].set_title(r"(e)  $\hat y=w^{(2)}_0+\sum_j w^{(2)}_j z_j$",
                       fontsize=16.2)

    J = joints()
    ax[1, 2].plot(x, yhat, color=OUTC, lw=2.6)
    for j, xc in enumerate(J):
        ax[1, 2].axvline(xc, color=BRANCH[j], lw=1.2, ls=":")
        ax[1, 2].plot([xc], [np.interp(xc, x, yhat)], marker="o", ms=7,
                      color=BRANCH[j], mec="white", mew=1.0, zorder=5)
    ax[1, 2].set(xlabel=r"$x$", ylabel=r"$\hat y$")
    ax[1, 2].set_title(r"(f)  one joint per unit, at $x=-w^{(1)}_{j0}/w^{(1)}_{j1}$",
                       fontsize=15.6)
    for axis in ax.ravel():
        style.square(axis)
    fig.tight_layout(w_pad=2.0, h_pad=2.4)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

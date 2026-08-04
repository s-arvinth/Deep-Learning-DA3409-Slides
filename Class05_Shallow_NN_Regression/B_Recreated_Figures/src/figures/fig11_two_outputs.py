#!/usr/bin/env python3
"""K = 2 outputs reading from one shared hidden layer."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import shallow, joints

NAME = "two_outputs"


def build():
    style.use()
    x = np.linspace(-1, 1, 1200)
    _, z, _ = shallow(x)
    A = np.array([-0.35,  1.6, -3.2,  2.4])
    Bv = np.array([ 0.30, -2.4,  3.0, -1.8])
    y1 = A[0] + (A[1:][:, None] * z).sum(0)
    y2 = Bv[0] + (Bv[1:][:, None] * z).sum(0)
    J = joints()

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 4.4), sharex=True)
    for axis, yv, tag, c in [(ax[0], y1, "a", OUTC), (ax[1], y2, "b", L2)]:
        axis.plot(x, yv, color=c, lw=2.6)
        for j, xc in enumerate(J):
            axis.axvline(xc, color=BRANCH[j], lw=1.1, ls=":")
            axis.plot([xc], [np.interp(xc, x, yv)], marker="o", ms=7,
                      color=BRANCH[j], mec="white", mew=1.0, zorder=5)
        axis.set(xlabel=r"$x$")
        axis.set_title(rf"({tag})  output $\hat y_{{{1 if tag=='a' else 2}}}"
                       rf"=w^{{(2)}}_{{{1 if tag=='a' else 2}0}}"
                       rf"+\sum_j w^{{(2)}}_{{{1 if tag=='a' else 2}j}}z_j$",
                       fontsize=15.0)
        style.square(axis)
    ax[0].set_ylabel(r"$\hat y$")
    fig.suptitle(r"shared hidden units $\Rightarrow$ joints at the same $x$",
                 fontsize=16.9, y=1.03)
    fig.tight_layout(w_pad=2.2)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

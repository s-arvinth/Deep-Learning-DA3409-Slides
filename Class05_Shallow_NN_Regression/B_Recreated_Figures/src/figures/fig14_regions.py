#!/usr/bin/env python3
"""Counting linear regions: Zaslavsky's bound, and the scalar-input case."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import relu
from math import comb

NAME = "regions"


def build():
    style.use()
    M = np.arange(1, 101)
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.4))

    cmap = plt.get_cmap(style.CAT)
    pal = cmap(np.linspace(0.05, 0.78, 4))
    for i, (D, mk) in enumerate([(1, "o"), (2, "s"), (5, "^"), (10, "D")]):
        reg = [sum(comb(int(m), j) for j in range(0, D + 1)) for m in M]
        ax[0].plot(M, reg, color=pal[i], lw=2.0, marker=mk, ms=6,
                   markevery=12, mew=0.8, mec="white",
                   label=rf"$D={D}$ inputs")
    ax[0].set(xscale="log", yscale="log", xlabel=r"hidden units  $M$",
              ylabel=r"maximum linear regions")
    ax[0].set_title(r"(a)  $\sum_{j=0}^{D}\binom{M}{j}$  (Zaslavsky, 1975)",
                    fontsize=15.6)
    ax[0].legend(loc="upper left", fontsize=13.7)

    x = np.linspace(0, 1, 1000)
    for M_, c, mk in zip([2, 5, 12], [L1, L2, OUTC], ["o", "s", "^"]):
        cuts = np.linspace(0, 1, M_ + 2)[1:-1]
        rng = np.random.default_rng(M_)
        yv = sum(rng.uniform(-1.6, 1.6) * relu(x - c) for c in cuts)
        yv = yv - yv.mean()
        ax[1].plot(x, yv, color=c, lw=2.0,
                   label=rf"$M={M_}$  ($M+1={M_+1}$ pieces)")
        ax[1].plot(cuts, np.interp(cuts, x, yv), ls="none", marker=mk, ms=6.5,
                   color=c, mec="white", mew=0.9)
    ax[1].legend(loc="lower left", fontsize=12.3, framealpha=.92,
                 facecolor="white", edgecolor="none")
    ax[1].set(xlabel=r"$x$", ylabel=r"$\hat y$")
    ax[1].set_title(r"(b)  $D=1$: markers are the joints, one per unit",
                    fontsize=15.6)
    for axis in ax:
        style.square(axis)
    fig.tight_layout(w_pad=2.4)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

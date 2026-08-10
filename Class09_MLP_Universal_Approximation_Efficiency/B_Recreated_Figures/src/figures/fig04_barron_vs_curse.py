#!/usr/bin/env python3
"""Two competing bounds: Barron's dimension-free 1/sqrt(M) upper bound,
and the DeVore-Howard-Micchelli W^{-s/D} lower bound for smoothness
classes.  Which one bites depends on the class the target lives in."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import barron_rate, sobolev_rate

NAME = "barron_vs_curse"


def build():
    style.use()
    W = np.logspace(1, 6, 200)
    cmap = plt.get_cmap(style.CAT)

    fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.6))

    ax[0].loglog(W, barron_rate(W), color=OUTC, lw=2.8,
                 label=r"Barron:  $C_f\,M^{-1/2}$  (any $D$)")
    for i, D in enumerate([1, 2, 5, 20]):
        ax[0].loglog(W, sobolev_rate(W, D, s=2.0), lw=2.0, ls="--",
                     color=cmap(0.05 + 0.22 * i),
                     label=rf"$W^{{-2/D}}$, $D={D}$")
    ax[0].legend(loc="lower left", fontsize=9.8)
    ax[0].set(xlabel=r"parameters $W$  (or units $M$)",
              ylabel=r"approximation error", ylim=(1e-6, 2))
    ax[0].set_title(r"(a)  a dimension-free rate, and one that is not",
                    fontsize=11.5)

    # how many parameters to reach a target accuracy
    eps = 1e-2
    Ds = np.arange(1, 21)
    need_sob = eps ** (-Ds / 2.0)
    ax[1].semilogy(Ds, np.full_like(Ds, 1.0 / eps ** 2, dtype=float),
                   color=OUTC, lw=2.8, marker="o", ms=6, markevery=2,
                   mec="white", mew=0.8,
                   label=r"Barron class:  $\varepsilon^{-2}$")
    ax[1].semilogy(Ds, need_sob, color=L1, lw=2.4, ls="--", marker="s",
                   ms=6, markevery=2, mec="white", mew=0.8,
                   label=r"$C^{2}$ ball:  $\varepsilon^{-D/2}$")
    ax[1].legend(loc="lower right", fontsize=10.5)
    ax[1].set(xlabel=r"input dimension $D$",
              ylabel=r"parameters needed for $\varepsilon=10^{-2}$")
    style.integer_ticks(ax[1], "x", nbins=5)
    ax[1].set_title(r"(b)  the same accuracy, as $D$ grows",
                    fontsize=11.5)

    for a in ax:
        style.square(a)
    fig.tight_layout(w_pad=2.4)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

#!/usr/bin/env python3
"""VC dimension of a ReLU network: Theta(W L log W).  Expressive power
and capacity to overfit grow together."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import vc_dim

NAME = "vc_growth"


def build():
    style.use()
    cmap = plt.get_cmap(style.CAT)
    W = np.logspace(1, 6, 200)

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.6))

    for i, (Lk, mk) in enumerate([(2, "o"), (5, "s"), (20, "^"), (100, "D")]):
        ax[0].loglog(W, vc_dim(W, Lk), color=cmap(0.05 + 0.22 * i), lw=2.2,
                     marker=mk, ms=6, markevery=40, mec="white", mew=0.8,
                     label=rf"$L={Lk}$")
    ax[0].loglog(W, W, color=GREY, lw=1.8, ls="--",
                 label=r"reference $W$")
    ax[0].legend(loc="upper left", fontsize=10.5)
    ax[0].set(xlabel=r"parameters $W$", ylabel=r"VC dimension")
    ax[0].set_title(r"(a)  $\mathrm{VCdim}=\Theta(WL\log W)$",
                    fontsize=12)

    # samples needed for a fixed generalisation gap, N ~ VCdim / gap^2
    gap = 0.1
    Ls = np.array([2, 5, 20, 100])
    Wfix = 1e5
    need = vc_dim(np.full_like(Ls, Wfix, dtype=float), Ls) / gap ** 2
    ax[1].bar(np.arange(len(Ls)), need, width=0.55,
              color=[cmap(0.05 + 0.22 * i) for i in range(len(Ls))],
              alpha=.85, edgecolor="white", lw=1.2)
    ax[1].set_yscale("log")
    ax[1].set_xticks(np.arange(len(Ls)))
    ax[1].set_xticklabels([rf"$L={v}$" for v in Ls])
    ax[1].set(xlabel=r"depth, at fixed $W=10^{5}$",
              ylabel=r"samples for a $0.1$ gap")
    ax[1].set_title(r"(b)  the worst-case sample bound", fontsize=12)

    for a in ax:
        style.square(a)
    fig.tight_layout(w_pad=2.4)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

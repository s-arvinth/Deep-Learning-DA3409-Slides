#!/usr/bin/env python3
"""sigma' <= 1/4, and what that costs over L layers."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import sigmoid, dsigmoid, dtanh

NAME = "saturation_bound"


def build():
    style.use()
    a = np.linspace(-8, 8, 900)
    fig, ax = plt.subplots(1, 3, figsize=(11.6, 4.4))

    ax[0].plot(a, dsigmoid(a), color=OUTC, lw=2.8,
               label=r"$\sigma'(a)=\sigma(1-\sigma)$")
    ax[0].axhline(0.25, color=L1, lw=2.0, ls="--", label=r"$1/4$")
    ax[0].plot([0], [0.25], marker="o", ms=9, color=L1, mec="white", mew=1.2)
    ax[0].legend(loc="upper right", fontsize=11)
    ax[0].set(xlabel=r"$a$", ylabel=r"$\sigma'(a)$", ylim=(0, 0.32))
    ax[0].set_title(r"(a)  the maximum is exactly $1/4$", fontsize=12)

    ax[1].plot(a, dtanh(a), color=L2, lw=2.8, label=r"$\tanh'(a)$")
    ax[1].axhline(1.0, color=L1, lw=2.0, ls="--", label=r"$1$")
    ax[1].legend(loc="upper right", fontsize=11)
    ax[1].set(xlabel=r"$a$", ylabel=r"$\tanh'(a)$", ylim=(0, 1.15))
    ax[1].set_title(r"(b)  $\tanh$ reaches $1$, but only at $a=0$",
                    fontsize=12)

    L = np.arange(1, 26)
    for f, c, nm, mk in [(0.25, OUTC, r"logistic:  $4^{-L}$", "o"),
                         (1.00, L2, r"$\tanh$ at best:  $1$", "s"),
                         (0.60, ACC, r"$\tanh$ typical:  $0.6^{L}$", "^")]:
        ax[2].semilogy(L, f ** L, color=c, lw=2.4, marker=mk, ms=6,
                       markevery=3, mec="white", mew=0.8, label=nm)
    ax[2].legend(loc="lower left", fontsize=10.5)
    ax[2].set(xlabel=r"layers $L$ traversed",
              ylabel=r"gradient factor $\prod_l h'$")
    ax[2].set_title(r"(c)  the product over depth", fontsize=12)
    style.integer_ticks(ax[2], "x", nbins=6)

    for a_ in ax:
        style.square(a_)
    fig.tight_layout(w_pad=2.2)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

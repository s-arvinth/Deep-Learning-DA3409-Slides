#!/usr/bin/env python3
"""Poggio et al. (2017): a COMPOSITIONAL target escapes the curse.

A generic smooth function on [0,1]^D needs O(eps^{-D/m}) parameters.  If
instead the target is built from a binary tree of two-variable functions,
a deep network matching that tree needs only O((D-1) eps^{-2/m}) --- the
dimension enters linearly, not in the exponent.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np


NAME = "compositional"


def generic(D, eps, m=2.0):
    return eps ** (-D / m)


def compositional(D, eps, m=2.0):
    return (D - 1.0) * eps ** (-2.0 / m)


def build():
    style.use()
    cmap = plt.get_cmap(style.CAT)
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.6))

    # (a) parameters against dimension, at a fixed accuracy
    D = np.arange(2, 26)
    eps = 0.1
    ax[0].semilogy(D, generic(D, eps), color=L1, lw=2.4, marker="s", ms=6,
                   markevery=3, mec="white", mew=0.8,
                   label=r"generic:  $\varepsilon^{-D/m}$")
    ax[0].semilogy(D, compositional(D, eps), color=OUTC, lw=2.6, marker="o",
                   ms=6, markevery=3, mec="white", mew=0.8,
                   label=r"compositional:  $(D-1)\varepsilon^{-2/m}$")
    ax[0].legend(loc="upper left", fontsize=11)
    ax[0].set(xlabel=r"input dimension $D$",
              ylabel=rf"parameters for $\varepsilon={eps:g}$")
    ax[0].set_title(r"(a)  the exponent moves out of $D$", fontsize=12.5)
    style.integer_ticks(ax[0], "x", nbins=5)

    # (b) parameters against accuracy, for three dimensions
    e = np.logspace(-3, -0.7, 200)
    for i, Dk in enumerate([4, 10, 20]):
        ax[1].loglog(e, generic(Dk, e), lw=2.0, ls="--",
                     color=cmap(0.05 + 0.30 * i),
                     label=rf"generic, $D={Dk}$")
    ax[1].loglog(e, compositional(20, e), color=OUTC, lw=2.8,
                 label=r"compositional, any $D\leq 20$")
    ax[1].legend(loc="upper right", fontsize=10)
    ax[1].set(xlabel=r"target accuracy $\varepsilon$",
              ylabel=r"parameters")
    ax[1].set_title(r"(b)  one curve, not a family", fontsize=12.5)

    for a in ax:
        style.square(a)
    fig.tight_layout(w_pad=2.4)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

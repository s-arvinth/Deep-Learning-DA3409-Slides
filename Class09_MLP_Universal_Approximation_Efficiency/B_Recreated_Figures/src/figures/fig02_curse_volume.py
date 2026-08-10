#!/usr/bin/env python3
"""Two more faces of the curse: volume concentrates in a thin outer
shell, and the mass of a Gaussian moves away from its mode."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from math import lgamma

NAME = "curse_volume"


def build():
    style.use()
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 4.4))
    cmap = plt.get_cmap(style.CAT)

    # (a) fraction of the volume of a unit ball within eps of the surface
    eps = np.linspace(0.0, 1.0, 400)
    for i, D in enumerate([1, 2, 5, 20]):
        ax[0].plot(eps, 1 - (1 - eps) ** D, lw=2.4,
                   color=cmap(0.05 + 0.23 * i), label=rf"$D={D}$")
    ax[0].legend(loc="lower right", fontsize=11.5)
    ax[0].set(xlabel=r"shell thickness $\epsilon$",
              ylabel=r"fraction of volume")
    ax[0].set_title(r"(a)  $1-(1-\epsilon)^{D}$: volume sits on the skin",
                    fontsize=11.5)

    # (b) radial density of a standard Gaussian in D dimensions
    r = np.linspace(0, 9, 900)
    for i, D in enumerate([1, 2, 5, 20]):
        logp = ((D - 1) * np.log(np.maximum(r, 1e-12)) - r ** 2 / 2
                - lgamma(D / 2) - (D / 2 - 1) * np.log(2))
        p = np.exp(logp)
        ax[1].plot(r, p / np.trapezoid(p, r), lw=2.4,
                   color=cmap(0.05 + 0.23 * i), label=rf"$D={D}$")
    ax[1].legend(loc="upper right", fontsize=11.5)
    ax[1].set(xlabel=r"radius $\|\mathbf{x}\|$",
              ylabel=r"radial density $p(r)$")
    ax[1].set_title(r"(b)  Gaussian mass leaves the mode", fontsize=11.5)

    # (c) samples needed to keep a fixed density on a grid
    D = np.arange(1, 21)
    for i, (n, mk) in enumerate([(5, "o"), (10, "s"), (20, "^")]):
        ax[2].semilogy(D, float(n) ** D, lw=2.2, marker=mk, ms=6,
                       markevery=2, mec="white", mew=0.8,
                       color=cmap(0.05 + 0.3 * i),
                       label=rf"${n}$ samples per axis")
    ax[2].axhline(1e12, color=GREY, lw=1.2, ls=":")
    ax[2].annotate("a trillion", xy=(1.5, 2e12), fontsize=11, color=GREY)
    ax[2].legend(loc="lower right", fontsize=11)
    ax[2].set(xlabel=r"input dimension $D$",
              ylabel=r"samples for the same density")
    style.integer_ticks(ax[2], "x", nbins=5)
    ax[2].set_title(r"(c)  data needed grows the same way", fontsize=11.5)

    for a in ax:
        style.square(a)
    fig.tight_layout(w_pad=2.2)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

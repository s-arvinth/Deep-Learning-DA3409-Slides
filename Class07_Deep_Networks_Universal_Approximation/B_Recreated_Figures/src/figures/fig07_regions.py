#!/usr/bin/env python3
"""Linear regions bought per parameter, and per layer at a fixed budget."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import (params_shallow, params_deep,
                           regions_shallow, regions_deep)

NAME = "regions"


def build():
    style.use()
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.6))
    pal = plt.get_cmap(style.CAT)(np.linspace(0.05, 0.75, 4))

    M = np.arange(2, 61)
    ax[0].plot(params_shallow(M), regions_shallow(M), color=pal[0],
               lw=2.2, marker="o", ms=6, markevery=8, mew=0.8, mec="w",
               label=r"shallow ($L=1$)")
    for i, (Lk, mk) in enumerate([(2, "s"), (3, "^"), (5, "D")], start=1):
        Mk = np.arange(2, 31)
        ax[0].plot(params_deep(Mk, Lk), regions_deep(Mk, Lk), color=pal[i],
                   lw=2.2, marker=mk, ms=6, markevery=6, mew=0.8, mec="w",
                   label=rf"deep, $L={Lk}$")
    ax[0].set(xscale="log", yscale="log",
              xlabel="number of parameters", ylabel="maximum linear regions")
    ax[0].set_title(r"(a)  regions bought per parameter", fontsize=12.5)
    ax[0].legend(loc="upper left", fontsize=11)

    Ls = np.arange(1, 11)
    pal2 = plt.get_cmap(style.CAT)(np.linspace(0.1, 0.7, 3))
    for (P, mk), c in zip([(200, "s"), (1000, "^"), (5000, "D")], pal2):
        best = []
        for Lk in Ls:
            fit = [m for m in range(1, 400)
                   if (params_shallow(m) if Lk == 1
                       else params_deep(m, Lk)) <= P]
            if not fit:
                best.append(np.nan); continue
            m = max(fit)
            best.append(regions_shallow(m) if Lk == 1
                        else regions_deep(m, Lk))
        ax[1].plot(Ls, best, color=c, lw=2.2, marker=mk, ms=7, mew=0.8,
                   mec="w", label=rf"budget $P={P}$")
    ax[1].set(yscale="log", xlabel=r"hidden layers $L$",
              ylabel="maximum linear regions")
    ax[1].set_title(r"(b)  one budget, spread over more layers",
                    fontsize=12.5)
    ax[1].legend(loc="lower right", fontsize=11)

    for a in ax:
        style.square(a)
    fig.tight_layout(w_pad=2.4)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

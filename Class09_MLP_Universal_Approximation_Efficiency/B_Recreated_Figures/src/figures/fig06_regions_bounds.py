#!/usr/bin/env python3
"""Linear regions against parameters: the shallow count, the Montufar
lower bound for deep networks, and a product upper bound."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import (n_params, regions_shallow,
                           regions_deep_lower, regions_deep_upper)

NAME = "regions_bounds"


def build():
    style.use()
    cmap = plt.get_cmap(style.CAT)
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.6))

    for k, D in enumerate([1, 5]):
        a_ = ax[k]
        Ms = np.arange(max(D, 2), 61)
        a_.loglog([n_params(m, 1, D) for m in Ms],
                  [regions_shallow(m, D) for m in Ms],
                  color=GREY, lw=2.6, marker="o", ms=6, markevery=9,
                  mec="white", mew=0.8, label=r"shallow ($L=1$)")
        for i, (Lk, mk) in enumerate([(2, "s"), (3, "^"), (5, "D")]):
            # only widths that are multiples of D: the Montufar bound is
            # stated for M >= D and its floor(M/D) makes intermediate
            # widths a distracting staircase
            Mk = np.arange(D, 41, D) if D > 1 else np.arange(2, 41, 2)
            lo = [regions_deep_lower(m, Lk, D) for m in Mk]
            a_.loglog([n_params(m, Lk, D) for m in Mk], lo,
                      color=cmap(0.05 + 0.26 * i), lw=2.2, marker=mk, ms=6,
                      markevery=8, mec="white", mew=0.8,
                      label=rf"deep $L={Lk}$, lower bound")
        Mu = np.arange(D, 41, D) if D > 1 else np.arange(2, 41, 2)
        a_.loglog([n_params(m, 3, D) for m in Mu],
                  [regions_deep_upper(m, 3, D) for m in Mu],
                  color=OUTC, lw=1.8, ls=":",
                  label=r"$L=3$, upper bound")
        a_.set(xlabel=r"parameters $W$",
               ylabel=r"linear regions")
        a_.set_title(rf"({'ab'[k]})  $D={D}$ input"
                     rf"{'s' if D > 1 else ''}", fontsize=12.5)
        a_.legend(loc="upper left", fontsize=9.5)
        style.square(a_)

    fig.tight_layout(w_pad=2.4)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

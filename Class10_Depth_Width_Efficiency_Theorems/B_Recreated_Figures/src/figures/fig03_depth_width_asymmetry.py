#!/usr/bin/env python3
"""The asymmetry that decides the lecture.

Emulating a DEEP network with a shallow one costs width EXPONENTIAL in
the depth (Telgarsky).  Emulating a WIDE shallow network with a narrow
one costs only POLYNOMIAL extra depth (Lu et al.), and for ReLU only a
LINEAR increase (Vardi et al.).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np


NAME = "depth_width_asymmetry"


def build():
    style.use()
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.6))

    # (a) cost of flattening a deep network
    L = np.arange(1, 17)
    ax[0].semilogy(L, 2.0 ** L - 1, color=OUTC, lw=2.6, marker="o", ms=6,
                   markevery=2, mec="white", mew=0.8,
                   label=r"width needed:  $2^{L}-1$")
    ax[0].semilogy(L, 2.0 * L, color=L1, lw=2.2, ls="--", marker="s", ms=6,
                   markevery=2, mec="white", mew=0.8,
                   label=r"the deep network:  $2L$ units")
    ax[0].legend(loc="upper left", fontsize=10.5)
    ax[0].set(xlabel=r"depth $L$ being flattened", ylabel=r"hidden units")
    ax[0].set_title(r"(a)  deep $\to$ shallow: \alert{exponential}"
                    .replace("\alert{", "").replace("}", "", 1),
                    fontsize=12.5)
    style.integer_ticks(ax[0], "x", nbins=6)

    # (b) cost of narrowing a wide network
    M = np.arange(2, 400)
    ax[1].plot(M, M ** 2 / 40.0, color=OUTC, lw=2.4, ls=":",
               label=r"polynomial (Lu et al.)")
    ax[1].plot(M, 2.0 * M, color=L2, lw=2.8,
               label=r"linear, ReLU (Vardi et al.)")
    ax[1].plot(M, np.full_like(M, 1.0, dtype=float), color=L1, lw=2.2,
               ls="--", label=r"the wide network:  depth $1$")
    ax[1].legend(loc="upper left", fontsize=10.5)
    ax[1].set(xlabel=r"width $M$ being narrowed", ylabel=r"depth needed",
              ylim=(0, 900))
    ax[1].set_title(r"(b)  wide $\to$ narrow: polynomial", fontsize=12.5)

    for a in ax:
        style.square(a)
    fig.tight_layout(w_pad=2.4)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

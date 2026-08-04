#!/usr/bin/env python3
"""A hierarchy built explicitly: parts, a motif, then modulation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import relu

NAME = "hierarchy"


def build():
    style.use()
    x = np.linspace(0, 1, 3000)

    def bump(c, w):
        return relu(1 - np.abs(x - c) / w)

    fig, ax = plt.subplots(1, 3, figsize=(11.4, 4.2))
    for c, col in zip([0.2, 0.5, 0.8], BRANCH):
        ax[0].plot(x, bump(c, 0.12), color=col, lw=2.4,
                   label=rf"$c={c}$")
    ax[0].legend(loc="upper right", fontsize=11)
    ax[0].set(xlabel=r"$x$", ylabel=r"$z^{(1)}_j$")
    ax[0].set_title(r"(a)  level 1: parts, $h(1-|x-c|/w)$", fontsize=12)

    motif = bump(0.2, .12) + 0.8 * bump(0.5, .12) + bump(0.8, .12)
    ax[1].plot(x, motif, color=GREY, lw=2.6)
    ax[1].set(xlabel=r"$x$", ylabel=r"$z^{(2)}_1$")
    ax[1].set_title(r"(b)  level 2: one motif from three parts",
                    fontsize=12)

    ax[2].plot(x, motif * (0.6 + 0.4 * np.sin(6 * np.pi * x)), color=OUTC,
               lw=2.6)
    ax[2].set(xlabel=r"$x$", ylabel=r"output")
    ax[2].set_title(r"(c)  level 3: the motif, modulated", fontsize=12)

    for a in ax:
        style.square(a)
    fig.tight_layout(w_pad=2.2)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

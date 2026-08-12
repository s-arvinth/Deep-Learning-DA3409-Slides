#!/usr/bin/env python3
"""A ReLU unit that switches off for every input can never switch back on."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import relu, dead_fraction

NAME = "dying_relu"


def build():
    style.use()
    fig, ax = plt.subplots(1, 3, figsize=(11.6, 4.4))

    x = np.linspace(-2, 2, 600)
    for b, c, nm in [(0.6, L2, r"alive:  $a=x+0.6$"),
                     (-0.4, ACC, r"partly:  $a=x-0.4$"),
                     (-2.6, OUTC, r"dead:  $a=x-2.6$")]:
        ax[0].plot(x, relu(x + b), color=c, lw=2.6, label=nm)
    ax[0].axhline(0, color=GREY, lw=0.7, ls=":")
    ax[0].legend(loc="upper left", fontsize=10.5)
    ax[0].set(xlabel=r"$x$", ylabel=r"$z=h(a)$")
    ax[0].set_title(r"(a)  three units over the data range", fontsize=12)

    ax[1].plot(x, (x + 0.6 > 0).astype(float), color=L2, lw=2.6,
               label=r"alive")
    ax[1].plot(x, (x - 0.4 > 0).astype(float), color=ACC, lw=2.6,
               label=r"partly")
    ax[1].plot(x, np.zeros_like(x), color=OUTC, lw=3.2, label=r"dead")
    ax[1].legend(loc="center left", fontsize=10.5)
    ax[1].set(xlabel=r"$x$", ylabel=r"$h'(a)$", ylim=(-0.12, 1.2))
    ax[1].set_title(r"(b)  its gradient is identically zero", fontsize=12)

    bs = np.linspace(0.0, -5.0, 26)
    ax[2].plot(-bs, [dead_fraction(v) for v in bs], color=OUTC, lw=2.6,
               marker="o", ms=6, markevery=3, mec="white", mew=0.8)
    ax[2].set(xlabel=r"how negative the bias is,  $-b$",
              ylabel=r"fraction of dead units")
    ax[2].set_title(r"(c)  a drifting bias switches units off for good",
                    fontsize=12)

    for a_ in ax:
        style.square(a_)
    fig.tight_layout(w_pad=2.2)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

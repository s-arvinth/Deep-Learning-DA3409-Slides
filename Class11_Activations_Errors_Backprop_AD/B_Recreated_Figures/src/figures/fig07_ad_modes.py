#!/usr/bin/env python3
"""Forward mode costs one sweep per INPUT; reverse mode one per OUTPUT."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np


NAME = "ad_modes"


def build():
    style.use()
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.6))
    cmap = plt.get_cmap(style.CAT)

    n = np.logspace(0, 6, 200)
    ax[0].loglog(n, n, color=L1, lw=2.6,
                 label=r"forward mode:  $D$ sweeps")
    ax[0].loglog(n, np.ones_like(n), color=OUTC, lw=2.6,
                 label=r"reverse mode:  $1$ sweep")
    ax[0].legend(loc="upper left", fontsize=11)
    ax[0].set(xlabel=r"inputs $D$  (one scalar output)",
              ylabel=r"sweeps of the graph", ylim=(0.5, 2e6))
    ax[0].set_title(r"(a)  training: $D=W$ is huge, $K=1$",
                    fontsize=12)

    k = np.logspace(0, 6, 200)
    ax[1].loglog(k, np.ones_like(k), color=L1, lw=2.6,
                 label=r"forward mode:  $1$ sweep")
    ax[1].loglog(k, k, color=OUTC, lw=2.6,
                 label=r"reverse mode:  $K$ sweeps")
    ax[1].legend(loc="upper left", fontsize=11)
    ax[1].set(xlabel=r"outputs $K$  (one scalar input)",
              ylabel=r"sweeps of the graph", ylim=(0.5, 2e6))
    ax[1].set_title(r"(b)  the mirror case", fontsize=12)

    for a_ in ax:
        style.square(a_)
    fig.suptitle(r"an error function has $W$ inputs and $1$ output "
                 r"$\Rightarrow$ reverse mode", fontsize=13, y=1.03)
    fig.tight_layout(w_pad=2.4)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

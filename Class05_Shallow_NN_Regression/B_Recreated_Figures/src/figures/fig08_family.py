#!/usr/bin/env python3
"""The family of functions a three-unit ReLU network can represent."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import shallow

NAME = "family"


def build():
    style.use()
    x = np.linspace(-1, 1, 1000)
    rng = np.random.default_rng(11)
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 4.2))
    for axis, tag in zip(ax, "abc"):
        for _ in range(4):
            w1 = np.stack([rng.uniform(-0.8, 0.8, 3),
                           rng.choice([-1, 1], 3) * rng.uniform(0.8, 2.2, 3)], 1)
            w2 = np.concatenate([[rng.uniform(-0.5, 0.5)],
                                 rng.uniform(-2.5, 2.5, 3)])
            axis.plot(x, shallow(x, w1, w2)[2], lw=2.0, alpha=.92)
        axis.axhline(0, color=GREY, lw=0.5, ls=":")
        axis.set(xlabel=r"$x$")
        axis.set_title(rf"({tag})  four random $\mathbf{{w}}$", fontsize=16.2)
        style.square(axis)
    ax[0].set_ylabel(r"$\hat y(x,\mathbf{w})$")
    fig.suptitle(r"$M=3$ units $\Rightarrow$ continuous, piecewise linear, "
                 r"at most $M+1=4$ pieces", fontsize=16.9, y=1.03)
    fig.tight_layout(w_pad=2.2)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

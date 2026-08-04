#!/usr/bin/env python3
"""The Bernoulli distribution: one parameter, the probability of y = 1."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np


NAME = "bernoulli"


def build():
    style.use()
    vals = [0.20, 0.50, 0.80]
    fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.9))
    for a_, p, c in zip(ax, vals, [L1, L2, OUTC]):
        a_.bar([0, 1], [1 - p, p], width=0.42, color=c, alpha=.85,
               edgecolor="white", lw=1.2)
        for xk, h in zip([0, 1], [1 - p, p]):
            a_.text(xk, h + 0.035, f"{h:.2f}", ha="center", fontsize=15.0,
                    color=c)
        a_.set(xticks=[0, 1], ylim=(0, 1.12), xlabel=r"$y$")
        a_.set_title(rf"$\hat y={p:.1f}$", fontsize=16.9)
        style.square(a_)
    ax[0].set_ylabel(r"$p(y\mid\hat y)$")
    fig.suptitle(r"$p(y\mid\hat y)=\hat y^{\,y}(1-\hat y)^{1-y}$,"
                 r"   $y\in\{0,1\}$", fontsize=17.6, y=1.04)
    fig.tight_layout(w_pad=2.2)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

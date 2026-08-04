#!/usr/bin/env python3
"""Why squared error stalls on a sigmoid output and cross-entropy does not."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import sigmoid, dsigmoid

NAME = "saturation"


def build():
    style.use()
    a = np.linspace(-8, 8, 900)
    yhat = sigmoid(a)
    y = 1.0                                   # true label is 1

    E_ce = -np.log(np.clip(yhat, 1e-12, 1))
    E_se = (yhat - y) ** 2
    g_ce = yhat - y
    g_se = (yhat - y) * dsigmoid(a)

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.5))
    ax[0].plot(a, E_ce, color=OUTC, lw=2.6, label=r"cross-entropy")
    ax[0].plot(a, E_se, color=L2, lw=2.6, ls="--", label=r"squared error")
    ax[0].legend(loc="upper right", fontsize=14.3)
    ax[0].set(xlabel=r"pre-activation $a$", ylabel=r"$E$ for one point",
              ylim=(-0.2, 6))
    ax[0].set_title(r"(a)  the error when the true label is $y=1$",
                    fontsize=15.6)

    ax[1].plot(a, np.abs(g_ce), color=OUTC, lw=2.6,
               label=r"$|\hat y-y|$   (cross-entropy)")
    ax[1].plot(a, np.abs(g_se), color=L2, lw=2.6, ls="--",
               label=r"$|(\hat y-y)\,\sigma'(a)|$   (squared)")
    ax[1].fill_between(a, 0, 1.05, where=a < -4, color=ACC, alpha=.18, lw=0)
    ax[1].annotate("confidently\nwrong", xy=(-7.7, 0.62), fontsize=14.3,
                   color=ACC)
    ax[1].legend(loc="upper right", fontsize=12.7)
    ax[1].set(xlabel=r"pre-activation $a$",
              ylabel=r"$|\partial E/\partial a|$", ylim=(0, 1.08))
    ax[1].set_title(r"(b)  squared error stops learning; "
                    r"cross-entropy does not", fontsize=15.0)
    for a_ in ax:
        style.square(a_)
    fig.tight_layout(w_pad=2.4)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

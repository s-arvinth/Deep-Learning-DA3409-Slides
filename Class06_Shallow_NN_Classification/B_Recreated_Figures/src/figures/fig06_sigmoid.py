#!/usr/bin/env python3
"""The logistic sigmoid and its derivative."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import sigmoid, dsigmoid

NAME = "sigmoid"


def build():
    style.use()
    a = np.linspace(-8, 8, 900)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.5))

    ax[0].plot(a, sigmoid(a), color=OUTC, lw=2.8,
               label=r"$\sigma(a)=1/(1+e^{-a})$")
    ax[0].axhline(0.5, color=GREY, lw=0.8, ls=":")
    ax[0].axvline(0.0, color=GREY, lw=0.8, ls=":")
    ax[0].axhline(1.0, color=GREY, lw=0.8, ls=":")
    ax[0].annotate(r"$\sigma(0)=\frac{1}{2}$", xy=(0.4, 0.53), fontsize=15.6,
                   color=GREY)
    ax[0].legend(loc="upper left", fontsize=14.3)
    ax[0].set(xlabel=r"$a$", ylabel=r"$\hat y=\sigma(a)$",
              ylim=(-0.06, 1.12))
    ax[0].set_title(r"(a)  maps $\mathbb{R}$ onto $(0,1)$", fontsize=16.2)

    ax[1].plot(a, dsigmoid(a), color=L2, lw=2.8,
               label=r"$\sigma'(a)=\sigma(a)(1-\sigma(a))$")
    ax[1].fill_between(a, 0, dsigmoid(a), where=np.abs(a) > 4,
                       color=ACC, alpha=.30, lw=0)
    ax[1].annotate("saturated:\n" r"$\sigma'\approx 0$", xy=(-7.6, 0.055),
                   fontsize=14.3, color=ACC)
    ax[1].annotate("saturated", xy=(4.7, 0.045), fontsize=14.3, color=ACC)
    ax[1].legend(loc="upper left", fontsize=13.7)
    ax[1].set(xlabel=r"$a$", ylabel=r"$\sigma'(a)$", ylim=(0, 0.32))
    ax[1].set_title(r"(b)  the gradient dies in the tails", fontsize=16.2)
    for a_ in ax:
        style.square(a_)
    fig.tight_layout(w_pad=2.4)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

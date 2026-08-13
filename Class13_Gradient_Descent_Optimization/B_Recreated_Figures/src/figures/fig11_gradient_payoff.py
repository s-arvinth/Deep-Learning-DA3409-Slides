"""Why gradients are worth computing at all: a counting argument.

Bishop & Bishop (2024), Section 7.2.1.  Near a minimum the error surface
is fixed by b and H, which carry W(W+3)/2 independent numbers.  Locating
the minimum should therefore need O(W^2) pieces of information.

  * Function values alone give one number per evaluation, so O(W^2)
    evaluations at O(W) each: O(W^3) work.
  * A gradient gives W numbers per evaluation, so O(W) evaluations, and
    backpropagation makes each one cost O(W): O(W^2) work.

The gap is a factor of W -- which for a network with a million weights
is a factor of a million.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S

NAME = "gradient_payoff"


def build():
    S.use()
    W = np.logspace(1, 8, 200)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2))

    ax = axes[0]
    ax.loglog(W, W ** 3, color=S.GREY, lw=2.4, label=r"values only: $O(W^3)$")
    ax.loglog(W, W ** 2, color=S.OUTC, lw=2.4, label=r"gradients: $O(W^2)$")
    for wv in (1e6,):
        ax.plot([wv, wv], [wv ** 2, wv ** 3], color=S.L2, lw=1.8,
                ls=(0, (4, 3)))
        ax.annotate(r"$\times\,10^{6}$", xy=(wv * 1.5, wv ** 2.5),
                    fontsize=15, color=S.L2)
    ax.set_xlabel("number of parameters $W$")
    ax.set_ylabel("steps to locate the minimum")
    ax.legend(loc="upper left", fontsize=13)
    ax.set_title("(a) the cost of not using gradients")
    S.square(ax)

    ax = axes[1]
    ax.loglog(W, np.ones_like(W), color=S.GREY, lw=2.4,
              label="one function value")
    ax.loglog(W, W, color=S.OUTC, lw=2.4, label=r"one gradient: $W$ numbers")
    ax.set_xlabel("number of parameters $W$")
    ax.set_ylabel("numbers returned per evaluation")
    ax.legend(loc="upper left", fontsize=13)
    ax.set_title("(b) information per evaluation")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

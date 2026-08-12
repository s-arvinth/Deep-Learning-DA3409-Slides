#!/usr/bin/env python3
"""An error function does not have to assume constant noise."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.data import heteroscedastic

NAME = "heteroscedastic"


def build():
    style.use()
    x, y, sd = heteroscedastic()
    xs = np.linspace(-1, 1, 400)
    mu = np.sin(2.2 * xs)
    sds = 0.05 + 0.45 * (xs + 1) / 2

    fig, ax = plt.subplots(1, 3, figsize=(11.6, 4.4))

    ax[0].scatter(x, y, s=22, color=L1, alpha=.5, lw=0)
    ax[0].plot(xs, mu, color=OUTC, lw=2.6, label=r"$\hat\mu(x)$")
    ax[0].fill_between(xs, mu - 2 * 0.28, mu + 2 * 0.28, color=GREY,
                       alpha=.22, lw=0, label=r"constant $\sigma$")
    ax[0].legend(loc="upper left", fontsize=10.5)
    ax[0].set(xlabel=r"$x$", ylabel=r"$y$")
    ax[0].set_title(r"(a)  one $\sigma$ for every $x$", fontsize=12)

    ax[1].scatter(x, y, s=22, color=L1, alpha=.5, lw=0)
    ax[1].plot(xs, mu, color=OUTC, lw=2.6)
    for k, al in [(1, .28), (2, .15)]:
        ax[1].fill_between(xs, mu - k * sds, mu + k * sds, color=OUTC,
                           alpha=al, lw=0)
    ax[1].set(xlabel=r"$x$", ylabel=r"$y$")
    ax[1].set_title(r"(b)  the network predicts $\sigma(x)$ too",
                    fontsize=12)

    ax[2].plot(xs, sds, color=L2, lw=2.8, label=r"$\hat\sigma(x)$")
    ax[2].axhline(0.28, color=GREY, lw=2.0, ls="--",
                  label=r"the single $\sigma$ fitted in (a)")
    ax[2].legend(loc="upper left", fontsize=10.5)
    ax[2].set(xlabel=r"$x$", ylabel=r"predicted standard deviation",
              ylim=(0, 0.62))
    ax[2].set_title(r"(c)  a second output head", fontsize=12)

    for a_ in ax:
        style.square(a_)
    fig.tight_layout(w_pad=2.2)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

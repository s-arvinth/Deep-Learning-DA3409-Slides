#!/usr/bin/env python3
"""Regression as maximum likelihood under a Gaussian -- the recipe, case 1."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.data import regression_cloud

NAME = "gaussian_ml"


def build():
    style.use()
    x, y, sd = regression_cloud()
    xs = np.linspace(0, 1, 400)
    mean = 0.35 + 1.25 * xs

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.6))
    ax[0].scatter(x, y, s=20, color=L1, alpha=.45, lw=0)
    for k, al in [(1, .26), (2, .13)]:
        ax[0].fill_between(xs, mean - k * sd, mean + k * sd, color=OUTC,
                           alpha=al, lw=0)
    ax[0].plot(xs, mean, color=OUTC, lw=2.6,
               label=r"$\hat y(x,\mathbf{w})$")
    for x0 in (0.25, 0.62):
        yy = np.linspace(mean[0] - 0.55, mean[-1] + 0.55, 300)
        pdf = np.exp(-((yy - (0.35 + 1.25 * x0)) ** 2) / (2 * sd ** 2))
        ax[0].plot(x0 + 0.16 * pdf, yy, color=L2, lw=1.9)
        ax[0].axvline(x0, color=GREY, lw=0.7, ls=":")
    ax[0].legend(loc="upper left", fontsize=14.3)
    ax[0].set(xlabel=r"$x$", ylabel=r"$y$", xlim=(0, 1.05),
              ylim=(-0.35, 2.35))
    ax[0].set_title(r"(a)  $p(y\mid x)=\mathcal{N}(y\mid\hat y,\sigma^2)$",
                    fontsize=15.6)

    yy = np.linspace(-1.2, 1.2, 500)
    for sd_, c in zip([0.20, 0.40, 0.70], [L1, L2, OUTC]):
        ax[1].plot(yy, np.exp(-yy ** 2 / (2 * sd_ ** 2)) /
                   (sd_ * np.sqrt(2 * np.pi)), lw=2.2, color=c,
                   label=rf"$\sigma={sd_:.2f}$")
    ax[1].legend(loc="upper right", fontsize=13.7)
    ax[1].set(xlabel=r"$y-\hat y$", ylabel="density")
    ax[1].set_title(r"(b)  $-\ln p \propto (y-\hat y)^2$: squared error",
                    fontsize=15.6)
    for a_ in ax:
        style.square(a_)
    fig.tight_layout(w_pad=2.4)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

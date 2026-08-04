#!/usr/bin/env python3
"""Least squares on 0/1 targets is dragged by distant correct points."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.data import two_gaussians
from common.models import fit_logistic, fit_least_squares

NAME = "lsq_vs_logistic"


def boundary(w, g, level=0.0):
    """Where w0 + w1 x1 + w2 x2 = level."""
    return (level - w[0] - w[1] * g) / w[2]


def build():
    style.use()
    rng = np.random.default_rng(5)
    X, y, _ = two_gaussians(N=70, sep=3.0,
                            cov=((0.75, 0.0), (0.0, 0.75)))
    extra = np.array([5.6, -4.4]) + 0.45 * rng.normal(size=(30, 2))

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.8))
    for k, (Xk, yk, ttl) in enumerate([
            (X, y, "(a)  well-separated data"),
            (np.vstack([X, extra]), np.r_[y, np.ones(len(extra))],
             "(b)  plus distant, correctly labelled points")]):
        wl = fit_least_squares(Xk, yk)
        wc = fit_logistic(Xk, yk)
        g = np.linspace(-6, 8, 300)
        ax[k].scatter(Xk[yk == 0, 0], Xk[yk == 0, 1], s=26, marker="o",
                      facecolor="white", edgecolor=L1, lw=1.2)
        ax[k].scatter(Xk[yk == 1, 0], Xk[yk == 1, 1], s=26, marker="s",
                      facecolor="white", edgecolor=OUTC, lw=1.2)
        ax[k].plot(g, boundary(wl, g, level=0.5), color=ACC, lw=2.4, ls="--",
                   label="least squares")
        ax[k].plot(g, boundary(wc, g), color=L2, lw=2.4,
                   label="logistic regression")
        ax[k].set(xlim=(-6, 8), ylim=(-6.5, 5.0), xlabel=r"$x_1$")
        ax[k].set_title(ttl, fontsize=16.2)
        style.square(ax[k])
    ax[0].set_ylabel(r"$x_2$")
    ax[1].legend(loc="lower left", fontsize=13.7, framealpha=.93,
                 facecolor="white", edgecolor="none")
    fig.tight_layout(w_pad=2.4)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

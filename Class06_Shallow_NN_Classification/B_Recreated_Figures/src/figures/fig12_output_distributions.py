#!/usr/bin/env python3
"""One recipe, many output domains: pick the distribution, get the loss."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from math import lgamma

NAME = "output_distributions"


def build():
    style.use()
    fig, ax = plt.subplots(2, 3, figsize=(11.4, 6.8))

    # --- univariate continuous: Gaussian
    t = np.linspace(-3.5, 3.5, 500)
    ax[0, 0].plot(t, np.exp(-t ** 2 / 2) / np.sqrt(2 * np.pi), color=L1, lw=2.6)
    ax[0, 0].set_title("real  " r"$y\in\mathbb{R}$" "\nGaussian  "
                       r"$\to$  squared error", fontsize=14.3)

    # --- binary: Bernoulli
    ax[0, 1].bar([0, 1], [0.3, 0.7], width=0.42, color=L2, alpha=.85,
                 edgecolor="white", lw=1.2)
    ax[0, 1].set_xticks([0, 1])
    ax[0, 1].set_title("binary  " r"$y\in\{0,1\}$" "\nBernoulli  "
                       r"$\to$  cross-entropy", fontsize=14.3)

    # --- categorical
    ax[0, 2].bar(np.arange(1, 6), [0.10, 0.32, 0.08, 0.38, 0.12], width=0.55,
                 color=OUTC, alpha=.85, edgecolor="white", lw=1.2)
    ax[0, 2].set_xticks(np.arange(1, 6))
    ax[0, 2].set_title("categorical  " r"$y\in\{1..K\}$" "\ncategorical  "
                       r"$\to$  cross-entropy", fontsize=14.3)

    # --- counts: Poisson
    k = np.arange(0, 13)
    lam = 3.4
    pmf = np.exp(k * np.log(lam) - lam - np.array([lgamma(i + 1) for i in k]))
    ax[1, 0].bar(k, pmf, width=0.6, color=ACC, alpha=.85, edgecolor="white",
                 lw=1.0)
    ax[1, 0].set_title("counts  " r"$y\in\{0,1,2,\dots\}$" "\nPoisson",
                       fontsize=14.3)

    # --- bounded: Beta
    u = np.linspace(1e-3, 1 - 1e-3, 500)
    for a_, b_, c in [(2, 5, L1), (5, 2, L2), (3, 3, OUTC)]:
        ax[1, 1].plot(u, u ** (a_ - 1) * (1 - u) ** (b_ - 1) *
                      np.exp(lgamma(a_ + b_) - lgamma(a_) - lgamma(b_)),
                      lw=2.2, color=c)
    ax[1, 1].set_title("bounded  " r"$y\in[0,1]$" "\nBeta", fontsize=14.3)

    # --- circular: von Mises
    th = np.linspace(-np.pi, np.pi, 500)
    for kap, c in [(0.8, L1), (3.0, L2), (9.0, OUTC)]:
        d = np.exp(kap * np.cos(th))
        ax[1, 2].plot(th, d / np.trapezoid(d, th), lw=2.2, color=c)
    ax[1, 2].set_xticks([-np.pi, 0, np.pi])
    ax[1, 2].set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    ax[1, 2].set_title("direction  " r"$y\in(-\pi,\pi]$" "\nvon Mises",
                       fontsize=14.3)

    for a_ in ax.ravel():
        a_.set_ylabel(r"$p(y)$")
        style.square(a_)
    fig.tight_layout(w_pad=2.0, h_pad=2.6)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

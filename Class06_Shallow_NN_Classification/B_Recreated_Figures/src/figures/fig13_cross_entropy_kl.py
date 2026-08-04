#!/usr/bin/env python3
"""Minimising the negative log-likelihood is minimising a KL divergence."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np


NAME = "cross_entropy_kl"


def build():
    style.use()
    rng = np.random.default_rng(2)
    y = np.linspace(-4, 5, 700)
    samples = rng.normal(1.0, 0.9, 26)

    def gauss(mu, sd):
        return np.exp(-((y - mu) ** 2) / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi))

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.6))

    ax[0].vlines(samples, 0, 0.085, color=L1, lw=2.2, alpha=.85,
                 label=r"empirical  $q(y)$")
    ax[0].scatter(samples, np.full_like(samples, 0.085), s=26,
                  color=L1, zorder=5)
    for mu, sd, c, ls in [(1.0, 0.9, OUTC, "-"), (-0.4, 1.7, ACC, "--")]:
        ax[0].plot(y, gauss(mu, sd), color=c, lw=2.4, ls=ls,
                   label=rf"model  $p(y\mid\hat y={mu:.1f})$")
    ax[0].legend(loc="upper right", fontsize=12.7)
    ax[0].set(xlabel=r"$y$", ylabel="density")
    ax[0].set_title(r"(a)  fit the model to the empirical distribution",
                    fontsize=15.0)

    mus = np.linspace(-2.5, 4.0, 400)
    nll = np.array([-np.log(np.exp(-((samples - m) ** 2) / (2 * 0.9 ** 2)) /
                            (0.9 * np.sqrt(2 * np.pi))).mean() for m in mus])
    ax[1].plot(mus, nll, color=OUTC, lw=2.6,
               label=r"$-\frac{1}{N}\sum_n \ln p(y_n\mid\hat y)$")
    ax[1].axvline(samples.mean(), color=L2, lw=1.6, ls="--",
                  label=r"minimum $=$ KL minimiser")
    ax[1].legend(loc="upper center", fontsize=12.7)
    ax[1].set(xlabel=r"model parameter $\hat y$",
              ylabel="negative log-likelihood")
    ax[1].set_title(r"(b)  $\mathrm{NLL} = H(q,p) = H(q)+\mathrm{KL}(q\|p)$",
                    fontsize=15.0)
    for a_ in ax:
        style.square(a_)
    fig.tight_layout(w_pad=2.4)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

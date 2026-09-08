"""The three sources of test error, one panel each.

Following the anatomy of Prince (2023), Fig. 8.5.

(a) Noise: even a model that reproduces h(x) exactly is scored against
    observations that carry N(0, sigma^2), so a floor of sigma^2 remains.

(b) Bias: the best fit available to a model of limited capacity still
    departs systematically from h(x). Here the capacity is K = 4 bumps
    and the departure is shaded.

(c) Variance: with a finite noisy sample we do not even recover that
    best fit. Twenty data sets give twenty different curves; their
    spread is the variance.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import truth, sample, piecewise_fit, SIGMA

NAME = "error_sources"

GRID = np.linspace(0.0, 1.0, 400)
K_LOW = 4


def build():
    S.use()
    rng = np.random.default_rng(3)
    h = truth(GRID)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))

    # ---- (a) noise ---------------------------------------------------
    ax = axes[0]
    ax.fill_between(GRID, h - 2 * SIGMA, h + 2 * SIGMA, color=S.GREY,
                    alpha=0.22, lw=0)
    ax.plot(GRID, h, color=S.GREY, lw=2.6, zorder=4)
    xs, ys = sample(30, rng)
    ax.plot(xs, ys, "o", ms=5.5, mfc="none", mec=S.L1, mew=1.3, zorder=5)
    ax.annotate(r"$\pm 2\sigma$", xy=(0.055, truth(0.055) + 2 * SIGMA + 0.10),
                color=S.GREY, fontsize=15)
    ax.set_title(r"(a) noise:  floor $\sigma^2$")

    # ---- (b) bias ----------------------------------------------------
    ax = axes[1]
    xb = np.linspace(0.0, 1.0, 4000)
    best = piecewise_fit(xb, truth(xb), K_LOW, grid=GRID)   # noiseless fit
    ax.fill_between(GRID, h, best, color=S.OUTC, alpha=0.20, lw=0)
    ax.plot(GRID, h, color=S.GREY, lw=2.6, label=r"$h(x)$", zorder=4)
    ax.plot(GRID, best, color=S.OUTC, lw=2.6, zorder=5,
            label=r"best fit, $K=%d$" % K_LOW)
    ax.legend(loc="upper right", fontsize=13)
    ax.set_title("(b) bias:  the best fit still misses")

    # ---- (c) variance ------------------------------------------------
    ax = axes[2]
    F = np.empty((20, len(GRID)))
    for l in range(20):
        x, y = sample(25, rng)
        F[l] = piecewise_fit(x, y, K_LOW, grid=GRID)
        ax.plot(GRID, F[l], color=S.L2, lw=1.0, alpha=0.45, zorder=3)
    ax.plot(GRID, F.mean(axis=0), color=S.L2, lw=2.8, zorder=5,
            label="ensemble mean")
    ax.plot(GRID, h, color=S.GREY, lw=2.6, zorder=4, label=r"$h(x)$")
    ax.legend(loc="upper right", fontsize=13)
    ax.set_title("(c) variance:  the fit moves with the sample")

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(-2.0, 2.0)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_box_aspect(0.78)

    fig.tight_layout(w_pad=1.5)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

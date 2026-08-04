#!/usr/bin/env python3
"""More hidden units -> closer approximation, plus the observed error rate."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.data import target_wave
from common.models import relu_design

NAME = "approximate"


def fit(x, target, M):
    cuts = np.linspace(0, 1, M + 2)[1:-1]
    A = relu_design(x, cuts)
    w, *_ = np.linalg.lstsq(A, target, rcond=None)
    return A @ w


def build():
    style.use()
    x = np.linspace(0, 1, 1400)
    target = target_wave(x)

    fig, ax = plt.subplots(2, 2, figsize=(9.2, 8.4))
    for axis, M, c in zip(ax.ravel()[:3], [2, 5, 20], [L1, L2, OUTC]):
        yh = fit(x, target, M)
        err = np.abs(target - yh).max()
        axis.plot(x, target, color=GREY, lw=2.0, ls="--", label=r"target $f(x)$")
        axis.plot(x, yh, color=c, lw=2.2, label=rf"$M={M}$ units")
        axis.legend(loc="upper right", fontsize=13.7)
        axis.set(xlabel=r"$x$", ylabel=r"$y$")
        axis.set_title(rf"$M={M}$:  $M+1={M+1}$ linear pieces,  "
                       rf"$\sup|f-\hat y|={err:.3f}$", fontsize=15.6)
        style.square(axis)

    Ms = np.unique(np.round(np.logspace(0, 2.1, 22)).astype(int))
    errs = [np.abs(target - fit(x, target, int(M))).max() for M in Ms]
    axis = ax[1, 1]
    axis.loglog(Ms, errs, color=OUTC, lw=2.0, marker="o", ms=6, mec="white",
                mew=0.8, label=r"$\sup_x|f(x)-\hat y(x)|$")
    axis.loglog(Ms, 0.9 * np.asarray(Ms, float) ** -2.0, color=GREY, lw=1.4,
                ls=":", label=r"reference slope $M^{-2}$")
    axis.legend(loc="lower left", fontsize=13.7)
    axis.set(xlabel=r"hidden units $M$", ylabel=r"maximum error")
    axis.set_title(r"error $\to 0$ as $M\to\infty$", fontsize=16.2)
    style.square(axis)
    fig.tight_layout(w_pad=2.4, h_pad=2.6)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

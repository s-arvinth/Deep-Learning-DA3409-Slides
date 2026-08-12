#!/usr/bin/env python3
"""When one output is not enough: a target with several valid answers.

Reading a one-to-one forward map backwards gives an inverse problem in
which one input has up to three correct outputs.  A network trained with
squared error predicts their average -- a value that is itself never
correct.  A mixture density network predicts a whole distribution.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.data import inverse_problem

NAME = "mixture_density"


def build():
    style.use()
    x, y = inverse_problem()
    ys = np.linspace(0, 1, 600)
    xs_curve = ys + 0.3 * np.sin(2 * np.pi * ys)

    fig, ax = plt.subplots(1, 3, figsize=(11.6, 4.4))

    ax[0].scatter(y, x, s=16, color=L1, alpha=.45, lw=0)
    ax[0].plot(ys, xs_curve, color=GREY, lw=2.2, ls="--")
    ax[0].set(xlabel=r"$y$", ylabel=r"$x$")
    ax[0].set_title(r"(a)  forward: one answer per input", fontsize=12)

    # the same data, axes swapped: now multi-valued
    ax[1].scatter(x, y, s=16, color=OUTC, alpha=.45, lw=0)
    x0 = 0.5
    ax[1].axvline(x0, color=L2, lw=2.0, ls=":")
    g = xs_curve - x0
    idx = np.where(np.sign(g[:-1]) != np.sign(g[1:]))[0]
    roots = ys[idx]                      # the genuine crossings
    ax[1].scatter(np.full_like(roots, x0), roots, s=90, marker="o",
                  facecolor="white", edgecolor=L2, lw=2.0, zorder=5)
    # a least-squares network predicts the conditional mean
    ax[1].scatter([x0], [roots.mean()], s=150, marker="X", color=ACC,
                  edgecolor="white", lw=1.2, zorder=6)
    ax[1].annotate("squared error\npredicts the mean", xy=(x0 + 0.05, roots.mean()),
                   color=ACC, fontsize=10.5)
    ax[1].set(xlabel=r"$x$", ylabel=r"$y$")
    ax[1].set_title(r"(b)  inverse: three answers at $x_0$", fontsize=12)

    # what a mixture predicts at x0
    t = np.linspace(0, 1, 600)
    dens = np.zeros_like(t)
    for c in roots:
        dens += np.exp(-((t - c) ** 2) / (2 * 0.045 ** 2)) / len(roots)
    dens /= np.trapezoid(dens, t)
    ax[2].plot(t, dens, color=OUTC, lw=2.8, label=r"$p(y\mid x_0)$")
    for c in roots:
        ax[2].axvline(c, color=L2, lw=1.2, ls=":")
    ax[2].axvline(roots.mean(), color=ACC, lw=2.0, ls="--",
                  label=r"the mean --- never correct")
    ax[2].legend(loc="upper left", fontsize=10)
    ax[2].set(xlabel=r"$y$", ylabel=r"density")
    ax[2].set_title(r"(c)  a mixture keeps all three", fontsize=12)

    for a_ in ax:
        style.square(a_)
    fig.tight_layout(w_pad=2.2)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

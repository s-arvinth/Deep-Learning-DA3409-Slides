#!/usr/bin/env python3
"""The constructive proof of universal approximation for D = 1, ReLU.

(a) Lemma 1: a scaled difference of two ReLUs is a ramp; as the gap
    shrinks it converges to a unit step.
(b) Lemma 2: any continuous f has an exact piecewise-linear interpolant
    on a grid, and that interpolant IS a shallow ReLU network.
(c) Step 3: uniform continuity turns mesh size into an error bound.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import relu

NAME = "ua_construction"


def f(x):
    return 0.65 * np.sin(2.4 * np.pi * x) * np.exp(-0.9 * x) + 0.45 * x


def build():
    style.use()
    x = np.linspace(0, 1, 2000)
    fig, ax = plt.subplots(1, 3, figsize=(12.0, 4.4))

    # ---- (a) ramp -> step ------------------------------------------
    a0 = 0.45
    cmap = plt.get_cmap(style.CAT)
    for i, d in enumerate([0.40, 0.20, 0.08, 0.02]):
        ramp = (relu(x - a0) - relu(x - a0 - d)) / d
        ax[0].plot(x, ramp, lw=2.2, color=cmap(0.06 + 0.22 * i),
                   label=rf"$\delta={d:g}$")
    ax[0].legend(loc="upper left", fontsize=13.7, title=r"gap $\delta$",
                 title_fontsize=13.7)
    ax[0].set(xlabel=r"$x$", ylabel=r"$g_{\delta}(x)$", ylim=(-0.08, 1.15))
    ax[0].set_title(r"(a)  $\frac{1}{\delta}[h(x-a)-h(x-a-\delta)]"
                    r"\to \mathbb{1}[x>a]$", fontsize=15.0)

    # ---- (b) the interpolant is a network ---------------------------
    M = 7
    knots = np.linspace(0, 1, M + 1)
    fk = f(knots)
    ax[1].plot(x, f(x), color=GREY, lw=2.2, ls="--", label=r"$f(x)$")
    ax[1].plot(knots, fk, color=OUTC, lw=2.2, marker="o", ms=7, mec="white",
               mew=1.0, label=r"$\hat y(x)$  ($M=7$ units)")
    for k in knots[1:-1]:
        ax[1].axvline(k, color=GREY, lw=0.6, ls=":")
    ax[1].legend(loc="lower left", fontsize=13.7, framealpha=.92,
                 facecolor="white", edgecolor="none")
    ax[1].set(xlabel=r"$x$", ylabel=r"$y$")
    ax[1].set_title(r"(b)  $\hat y=f(x_0)+s_1(x-x_0)+\sum_j(s_{j+1}-s_j)h(x-x_j)$",
                    fontsize=13.7)

    # ---- (c) mesh size controls the error ---------------------------
    for i, Mi in enumerate([3, 6, 12]):
        kn = np.linspace(0, 1, Mi + 1)
        yh = np.interp(x, kn, f(kn))
        e = np.abs(f(x) - yh)
        ax[2].plot(x, e, lw=2.0, color=cmap(0.06 + 0.28 * i),
                   label=rf"$M={Mi}$:  $\sup|f-\hat y|={e.max():.3f}$")
    ax[2].legend(loc="upper right", fontsize=13.0, framealpha=.92,
                 facecolor="white", edgecolor="none")
    ax[2].set(xlabel=r"$x$", ylabel=r"$|f(x)-\hat y(x)|$")
    ax[2].set_title(r"(c)  mesh $<\delta\,\Rightarrow\,"
                    r"\sup|f-\hat y|\leq\varepsilon$", fontsize=15.0)

    for axis in ax:
        style.square(axis)
    fig.tight_layout(w_pad=2.4)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

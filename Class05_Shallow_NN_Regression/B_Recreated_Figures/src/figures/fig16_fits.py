#!/usr/bin/env python3
"""A shallow ReLU network fitted to noisy samples of two functions."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import relu_design

NAME = "fits"


def build():
    style.use()
    rng = np.random.default_rng(7)
    x = np.sort(rng.uniform(-1, 1, 60))
    xs = np.linspace(-1, 1, 800)
    cases = [(lambda z: z ** 2,                  r"$f(x)=x^2$"),
             (lambda z: np.sin(2 * np.pi * z),   r"$f(x)=\sin(2\pi x)$")]

    fig, ax = plt.subplots(1, 2, figsize=(8.8, 4.4))
    for axis, (fn, nm), tag in zip(ax, cases, "ab"):
        y = fn(x) + 0.10 * rng.normal(size=len(x))
        cuts = np.linspace(-1, 1, 10)[1:-1]
        A = relu_design(x, cuts)
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        axis.scatter(x, y, s=34, facecolor="white", edgecolor=L1, lw=1.4,
                     zorder=3, label=r"data $(x_n,y_n)$")
        axis.plot(xs, fn(xs), color=GREY, lw=1.8, ls="--", label=r"truth")
        axis.plot(xs, relu_design(xs, cuts) @ w, color=OUTC, lw=2.4,
                  label=r"network $\hat y$")
        axis.set(xlabel=r"$x$")
        axis.set_title(rf"({tag})  {nm},  $M=8$ units", fontsize=12)
        style.square(axis)
    ax[0].set_ylabel(r"$y$")
    h, lb = ax[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=3, fontsize=11.5,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(w_pad=2.4)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

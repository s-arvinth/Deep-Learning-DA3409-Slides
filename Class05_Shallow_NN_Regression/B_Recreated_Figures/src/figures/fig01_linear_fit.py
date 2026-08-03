#!/usr/bin/env python3
"""Least-squares line through noisy data, residuals drawn explicitly."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.data import linear_data

NAME = "linear_fit"


def build():
    style.use()
    x, y = linear_data()
    A = np.stack([np.ones_like(x), x], 1)
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    xs = np.linspace(0, 1, 200)

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for xi, yi in zip(x, y):
        ax.plot([xi, xi], [yi, w[0] + w[1] * xi], color=GREY, lw=1.0,
                ls=":", zorder=1)
    ax.plot([], [], color=GREY, lw=1.0, ls=":",
            label=r"residual $y_n-\hat y(x_n)$")
    ax.scatter(x, y, s=52, marker="o", facecolor="white", edgecolor=L1,
               lw=1.7, zorder=3, label=r"data  $(x_n,\,y_n)$")
    ax.plot(xs, w[0] + w[1] * xs, color=OUTC, lw=2.4, zorder=2,
            label=r"$\hat y(x,\mathbf{w})=w_0+w_1x$")
    ax.legend(loc="upper left", fontsize=10.5, handlelength=1.6)
    ax.set(xlabel=r"input  $x$", ylabel=r"target  $y$",
           title=rf"minimising $\sum_n(y_n-\hat y_n)^2$:  "
                 rf"$w_0={w[0]:.2f}$,  $w_1={w[1]:.2f}$")
    ax.title.set_fontsize(12.5)
    style.square(ax)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

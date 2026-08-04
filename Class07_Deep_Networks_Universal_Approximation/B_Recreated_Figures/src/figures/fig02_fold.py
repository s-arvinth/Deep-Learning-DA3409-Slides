#!/usr/bin/env python3
"""Folding: each monotone branch of f1 carries its own copy of f2."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import f1, f2

NAME = "fold"


def build():
    style.use()
    x = np.linspace(-1, 1, 3000)
    y = f1(x)
    d = np.diff(y)
    turn = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0] + 1
    edges = [0] + list(turn) + [len(x) - 1]

    fig, ax = plt.subplots(1, 3, figsize=(11.4, 4.2))
    for i in range(len(edges) - 1):
        s = slice(edges[i], edges[i + 1] + 1)
        ax[0].plot(x[s], y[s], color=BRANCH[i % 4], lw=2.6)
        ax[2].plot(x[s], f2(y)[s], color=BRANCH[i % 4], lw=2.4)
    ax[0].set(xlabel=r"input $x$", ylabel=r"$u=f_1(x)$")
    ax[0].set_title(r"(a)  $f_1$ folds $x$ into three branches",
                    fontsize=12.5)

    yy = np.linspace(y.min(), y.max(), 2000)
    ax[1].plot(yy, f2(yy), color=GREY, lw=2.6)
    ax[1].set(xlabel=r"folded coordinate $u$", ylabel=r"$f_2(u)$")
    ax[1].set_title(r"(b)  $f_2$ sees only $u$", fontsize=12.5)

    ax[2].set(xlabel=r"input $x$", ylabel=r"$f_2(f_1(x))$")
    ax[2].set_title(r"(c)  one copy of $f_2$ per branch", fontsize=12.5)
    for i in turn:
        ax[0].axvline(x[i], color=GREY, lw=0.9, ls=":")
        ax[2].axvline(x[i], color=GREY, lw=0.9, ls=":")
    for a in ax:
        style.square(a)
    fig.tight_layout(w_pad=2.2)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

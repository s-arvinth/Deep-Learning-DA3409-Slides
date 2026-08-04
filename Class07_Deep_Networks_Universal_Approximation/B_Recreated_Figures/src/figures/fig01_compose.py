#!/usr/bin/env python3
"""Feeding one shallow network into another."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import f1, f2

NAME = "compose"


def build():
    style.use()
    x = np.linspace(-1, 1, 2000)
    y = f1(x)
    yy = np.linspace(y.min(), y.max(), 2000)

    fig, ax = plt.subplots(1, 3, figsize=(11.4, 4.2))
    ax[0].plot(x, y, color=L1, lw=2.6)
    ax[0].set(xlabel=r"input $x$", ylabel=r"$f_1(x)$")
    ax[0].set_title(r"(a)  network 1: 4 pieces", fontsize=13)

    ax[1].plot(yy, f2(yy), color=L2, lw=2.6)
    ax[1].set(xlabel=r"input $u$", ylabel=r"$f_2(u)$")
    ax[1].set_title(r"(b)  network 2: 4 pieces", fontsize=13)

    ax[2].plot(x, f2(y), color=OUTC, lw=2.6)
    ax[2].set(xlabel=r"input $x$", ylabel=r"$f_2(f_1(x))$")
    ax[2].set_title(r"(c)  composition: up to $4\times 4$", fontsize=13)

    for a in ax:
        a.axhline(0, color=GREY, lw=0.7, ls=":")
        style.square(a)
    fig.tight_layout(w_pad=2.2)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

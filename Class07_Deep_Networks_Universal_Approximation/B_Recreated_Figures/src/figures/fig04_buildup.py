#!/usr/bin/env python3
"""The two-layer computation, stage by stage."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import layer, f2, W1_1, W2_1

NAME = "buildup"


def build():
    style.use()
    x = np.linspace(-1, 1, 2000)
    a1, z1, out1 = layer(x, W1_1, W2_1)

    fig, ax = plt.subplots(2, 3, figsize=(11.6, 7.6))
    for k in range(3):
        b, w = W1_1[k]
        ax[0, k].plot(x, a1[k], color=GREY, lw=1.3, ls="--",
                      label=r"pre-activation $a^{(1)}_j$")
        ax[0, k].plot(x, z1[k], color=L1, lw=2.4,
                      label=r"$z^{(1)}_j=h(a^{(1)}_j)$")
        ax[0, k].axhline(0, color=GREY, lw=0.6, ls=":")
        ax[0, k].set(xlabel=r"$x$", ylabel=rf"$z^{{(1)}}_{k+1}$")
        ax[0, k].set_title(rf"({chr(97+k)})  $z^{{(1)}}_{k+1}"
                           rf"=h(x{b:+.1f})$", fontsize=12.5)
    ax[0, 0].legend(loc="upper left", fontsize=10.5)

    for k in range(3):
        ax[1, 0].plot(x, W2_1[k + 1] * z1[k], color=BRANCH[k], lw=2.2,
                      label=rf"$w^{{(2)}}_{k+1}z^{{(1)}}_{k+1}$")
    ax[1, 0].axhline(0, color=GREY, lw=0.6, ls=":")
    ax[1, 0].legend(fontsize=11)
    ax[1, 0].set(xlabel=r"$x$", ylabel=r"$w^{(2)}_j z^{(1)}_j$")
    ax[1, 0].set_title(r"(d)  weighted units of layer 1", fontsize=12.5)

    ax[1, 1].plot(x, out1, color=L1, lw=2.6)
    ax[1, 1].axhline(0, color=GREY, lw=0.6, ls=":")
    ax[1, 1].set(xlabel=r"$x$", ylabel=r"$f_1(x)$")
    ax[1, 1].set_title(r"(e)  summed: $f_1(x)$", fontsize=12.5)

    ax[1, 2].plot(x, f2(out1), color=OUTC, lw=2.6)
    ax[1, 2].axhline(0, color=GREY, lw=0.6, ls=":")
    ax[1, 2].set(xlabel=r"$x$", ylabel=r"$f_2(f_1(x))$")
    ax[1, 2].set_title(r"(f)  pushed through layer 2", fontsize=12.5)

    for a_ in ax.ravel():
        style.square(a_)
    fig.tight_layout(w_pad=2.0, h_pad=2.4)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

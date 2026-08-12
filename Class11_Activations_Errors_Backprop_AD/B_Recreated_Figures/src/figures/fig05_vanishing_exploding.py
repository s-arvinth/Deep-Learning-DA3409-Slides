#!/usr/bin/env python3
"""How the backpropagated signal changes size as it crosses layers."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import layer_gradient_norms

NAME = "vanishing_exploding"


def build():
    style.use()
    L = 20
    fig, ax = plt.subplots(1, 3, figsize=(11.6, 4.4))
    cmap = plt.get_cmap(style.CAT)
    layers = np.arange(1, L + 1)

    for k, (act, ttl) in enumerate([("tanh", r"(a)  $\tanh$"),
                                    ("relu", r"(b)  ReLU")]):
        for i, gain in enumerate([0.7, 1.0, 1.4]):
            n = layer_gradient_norms(L=L, gain=gain, act=act)
            ax[k].semilogy(layers, n / n[-1], lw=2.4,
                           color=cmap(0.05 + 0.30 * i), marker="o", ms=5,
                           markevery=3, mec="white", mew=0.7,
                           label=rf"gain ${gain}$")
        ax[k].axhline(1.0, color=GREY, lw=1.0, ls="--")
        ax[k].legend(loc="upper right", fontsize=10.5)
        ax[k].set(xlabel=r"layer $l$ (1 = input side)",
                  ylabel=r"$\|\delta^{(l)}\|$, relative to the top")
        ax[k].set_title(ttl, fontsize=12.5)
        style.integer_ticks(ax[k], "x", nbins=5)

    gains = np.linspace(0.5, 1.7, 22)
    for act, c, mk in [("tanh", L2, "s"), ("relu", OUTC, "o")]:
        ratio = [layer_gradient_norms(L=L, gain=g, act=act)[0] /
                 layer_gradient_norms(L=L, gain=g, act=act)[-1] for g in gains]
        ax[2].semilogy(gains, ratio, color=c, lw=2.4, marker=mk, ms=6,
                       markevery=3, mec="white", mew=0.8, label=act)
    ax[2].axhline(1.0, color=GREY, lw=1.2, ls="--")
    ax[2].legend(loc="upper left", fontsize=11)
    ax[2].set(xlabel=r"initialisation gain",
              ylabel=r"bottom $/$ top gradient norm")
    ax[2].set_title(r"(c)  a narrow band survives $L=20$", fontsize=12)
    style.square(ax[2])
    fig.tight_layout(w_pad=2.2)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

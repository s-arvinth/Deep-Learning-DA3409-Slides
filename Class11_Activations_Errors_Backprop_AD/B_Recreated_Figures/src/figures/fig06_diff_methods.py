#!/usr/bin/env python3
"""Why numerical differentiation is not an option.

(a) forward and central differences: truncation error falls with the step,
rounding error rises, and the best attainable accuracy is poor.
(b) the cost of one gradient, as the parameter count grows.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np


NAME = "diff_methods"


def f(w):
    return np.sin(w) * np.exp(-0.3 * w)


def df(w):
    return np.cos(w) * np.exp(-0.3 * w) - 0.3 * np.sin(w) * np.exp(-0.3 * w)


def build():
    style.use()
    w0 = 0.8
    eps = np.logspace(-16, -1, 220)
    fwd = np.abs((f(w0 + eps) - f(w0)) / eps - df(w0))
    ctr = np.abs((f(w0 + eps) - f(w0 - eps)) / (2 * eps) - df(w0))

    fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.6))
    ax[0].loglog(eps, np.maximum(fwd, 1e-18), color=OUTC, lw=2.2,
                 label=r"forward:  $O(\epsilon)$")
    ax[0].loglog(eps, np.maximum(ctr, 1e-18), color=L2, lw=2.2,
                 label=r"central:  $O(\epsilon^{2})$")
    ax[0].legend(loc="lower left", fontsize=11)
    ax[0].set(xlabel=r"step $\epsilon$", ylabel=r"error in $dE/dw$")
    ax[0].set_title(r"(a)  truncation down, rounding up", fontsize=12.5)

    W = np.logspace(1, 6, 200)
    ax[1].loglog(W, W ** 2, color=OUTC, lw=2.6,
                 label=r"finite differences:  $O(W^{2})$")
    ax[1].loglog(W, 4 * W, color=L2, lw=2.6,
                 label=r"backpropagation:  $O(W)$")
    ax[1].legend(loc="upper left", fontsize=11)
    ax[1].set(xlabel=r"parameters $W$",
              ylabel=r"operations for one $\nabla E$")
    ax[1].set_title(r"(b)  and it costs $W$ times more", fontsize=12.5)

    for a_ in ax:
        style.square(a_)
    fig.tight_layout(w_pad=2.4)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

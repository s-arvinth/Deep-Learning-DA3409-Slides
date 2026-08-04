#!/usr/bin/env python3
"""Hidden units act as basis functions that ADAPT to the target."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np


NAME = "learned_basis"


def build():
    style.use()
    x = np.linspace(-1, 1, 600)
    targets = [(r"$f(x)=x^2$",          x ** 2),
               (r"$f(x)=\sin(2\pi x)$", np.sin(2 * np.pi * x)),
               (r"$f(x)=|x|$",          np.abs(x)),
               (r"$f(x)=\mathbb{1}[x>0]$", (x > 0).astype(float))]

    def fit(target, M=6):
        cuts = np.linspace(-1, 1, M + 2)[1:-1]
        A = np.stack([np.tanh(4 * (x - c)) for c in cuts] +
                     [np.ones_like(x)], 1)
        w, *_ = np.linalg.lstsq(A, target, rcond=None)
        return A, w

    fig, ax = plt.subplots(1, 4, figsize=(12.4, 3.9))
    for axis, (nm, tgt) in zip(ax, targets):
        A, w = fit(tgt)
        for j in range(A.shape[1] - 1):
            axis.plot(x, w[j] * A[:, j], color=L2, lw=1.2, ls="--", alpha=.8)
        axis.plot([], [], color=L2, lw=1.2, ls="--",
                  label=r"$w^{(2)}_j z_j$")
        axis.plot(x, tgt, color=GREY, lw=1.8, ls=":", label=r"target")
        axis.plot(x, A @ w, color=OUTC, lw=2.4, label=r"$\hat y$")
        axis.set(xlabel=r"$x$")
        axis.set_title(nm, fontsize=15.6)
        style.square(axis)
    ax[0].set_ylabel(r"$y$")
    h, lb = ax[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=3, fontsize=15.0,
               frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(r"same architecture ($M=6$, $\tanh$), four targets: "
                 r"the units rearrange themselves", fontsize=16.2, y=1.04)
    fig.tight_layout(w_pad=1.8)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

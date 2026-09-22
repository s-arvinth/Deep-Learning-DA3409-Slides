"""Why a repeated linear map vanishes or explodes.

(a) ||W^t h_0|| against t for a random 20 x 20 matrix rescaled to
    spectral radius 0.9, 1.0 and 1.1 -- Goodfellow Eqs. 10.36-10.39:
    the eigenvalues are raised to the power t, so every component
    decays or explodes geometrically and only the largest eigenvector
    survives.
(b) The layout of Goodfellow Fig. 10.15: a random linear-tanh layer
    composed 1, 2, 3, 5 and 8 times, seen along one line through a
    100-dimensional state and projected to one number.  The
    composition is flat almost everywhere and steep in a few places,
    and the more times it is composed the more so.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import power_iteration, linear_tanh_composition

NAME = "power_iteration"


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    for rho, col in ((0.9, S.L2), (1.0, S.GREY), (1.1, S.OUTC)):
        n = power_iteration(rho, 60)
        ax.semilogy(n, color=col, lw=2.4, label="spectral radius $%.1f$" % rho)
        print("rho %.1f: norm after 60 steps %.2e" % (rho, n[-1]))
    ax.set_xlabel("steps $t$"); ax.set_ylabel("$\\|\\mathbf{U}^t \\mathbf{h}_0\\|$")
    ax.legend(fontsize=11); ax.set_title("(a) the power method, three radii")
    S.square(ax)
    ax = axes[1]
    xs, outs = linear_tanh_composition(8)
    cols = plt.get_cmap(S.CAT)(np.linspace(0.05, 0.85, 5))
    for k, col in zip((0, 1, 2, 4, 7), cols):
        ax.plot(xs, outs[k], color=col, lw=2.0, label="%d" % (k + 1))
    ax.set_xlabel("coordinate along a line in the state"); ax.set_ylabel("projected output")
    ax.legend(title="compositions", fontsize=10.5, title_fontsize=10.5, loc="lower right")
    ax.set_title("(b) a linear--tanh layer composed $t$ times")
    S.square(ax)
    fig.tight_layout(w_pad=2.0)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

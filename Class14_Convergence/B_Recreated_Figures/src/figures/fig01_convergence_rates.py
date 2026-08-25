"""Three rates of convergence, on the axes that make them distinguishable.

Plotted as the error above the optimum against the iteration count, with
a logarithmic vertical axis. On these axes a geometric (linear) rate is
a straight line, a sublinear rate bends the wrong way, and a quadratic
rate falls off a cliff.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S

NAME = "convergence_rates"


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2))

    T = np.arange(1, 61)

    ax = axes[0]
    ax.semilogy(T, 1.0 / T, color=S.ACC, lw=2.4,
                label=r"sublinear,  $O(1/T)$")
    ax.semilogy(T, 0.88 ** T, color=S.L1, lw=2.4,
                label=r"linear,  $\rho^{T}$ with $\rho = 0.88$")
    ax.semilogy(T, 0.60 ** T, color=S.L2, lw=2.4,
                label=r"linear,  $\rho = 0.60$")
    ax.set_xlabel("iteration $T$")
    ax.set_ylabel(r"$E(w^{(T)}) - E(w^\star)$")
    ax.set_ylim(1e-12, 3)
    ax.legend(loc="upper right", fontsize=12.5)
    ax.set_title("(a) a straight line means geometric")
    S.square(ax)

    ax = axes[1]
    rho = np.linspace(0.05, 0.985, 400)
    ax.plot(rho, np.log(1e-6) / np.log(rho), color=S.OUTC, lw=2.6)
    for r, col in ((0.60, S.L2), (0.88, S.L1)):
        n = np.log(1e-6) / np.log(r)
        ax.plot([r], [n], "o", ms=9, mfc="white", mec=col, mew=2.2,
                zorder=5)
        ax.annotate(r"$\rho = %.2f$: %d steps" % (r, round(n)),
                    xy=(r, n), xytext=(r - 0.05, n + 22), fontsize=13.5,
                    color=col, ha="right")
    ax.set_xlabel(r"contraction factor $\rho$ per step")
    ax.set_ylabel(r"steps to reach $10^{-6}$")
    ax.set_xlim(0.05, 1.0)
    ax.set_ylim(0, 340)
    ax.set_title(r"(b) the cost of a $\rho$ close to one")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

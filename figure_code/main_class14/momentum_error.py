"""momentum_error — what the three valley trajectories (descent, momentum, Nesterov) cost, and why

Used in: Class 14 (B), slide(s): "What that costs in iterations"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python momentum_error.py            -> output/momentum_error.pdf
         python momentum_error.py --out DIR  -> DIR/momentum_error.pdf
Ported from: Class14_Convergence/B_Recreated_Figures/src/figures/fig15_momentum_error.py

Panel (a): the error of the same three runs, on a logarithmic axis. A
geometric rate is a straight line, so the slope is log rho: momentum and
Nesterov descend visibly steeper lines than gradient descent.

Panel (b): the same statement as a function of the condition number.
Plotting 1 - rho on log-log turns the two rates into straight lines of
slope -1 and -1/2, which is the whole content of Polyak's result:
kappa becomes sqrt(kappa).
"""
import numpy as np
import matplotlib.pyplot as plt
import common as S
from models import quad_E

from nesterov_path import LAM, runs     # was fig07_momentum_paths

NAME = "momentum_error"


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))

    ax = axes[0]
    for key in ("gd", "mom", "nes"):
        title, sub, p, col = runs()[key]
        E = quad_E(p[:, 0], p[:, 1], lam=LAM)
        ax.semilogy(np.maximum(E, 1e-14), color=col, lw=2.3, label=title)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$E(\mathbf{w})$")
    ax.set_ylim(1e-8, 1e2)
    ax.set_xlim(0, 46)
    ax.legend(loc="upper right", fontsize=13)
    ax.set_title("(a) the three runs, by error")
    S.square(ax)

    ax = axes[1]
    k = np.logspace(0.02, 4, 300)
    ax.loglog(k, 2.0 / (k + 1.0), color=S.GREY, lw=2.4,
              label=r"gradient descent:  $1-\rho = 2/(\kappa+1)$")
    ax.loglog(k, 2.0 / (np.sqrt(k) + 1.0), color=S.OUTC, lw=2.4,
              label=r"momentum:  $1-\rho = 2/(\sqrt{\kappa}+1)$")
    ax.axvline(1e4, color=S.GREY, lw=0.9, ls=(0, (3, 3)), alpha=0.6)
    ax.annotate("", xy=(1e4, 2.0e-4), xytext=(1e4, 2.0e-2),
                arrowprops=dict(arrowstyle="<|-|>", color=S.L1, lw=1.6))
    ax.annotate(r"at $\kappa = 10^{4}$," "\n" r"$100\times$ fewer steps",
                xy=(7.5e3, 2.1e-3), fontsize=12.5, color=S.L1,
                ha="right", va="center")
    ax.set_xlabel(r"condition number $\kappa$")
    ax.set_ylabel(r"$1-\rho$   (larger is faster)")
    ax.set_ylim(1e-4, 2)
    ax.legend(loc="lower left", fontsize=12.5)
    ax.set_title(r"(b) $\kappa$ becomes $\sqrt{\kappa}$")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

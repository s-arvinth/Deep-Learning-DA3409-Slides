"""Each eigen-direction contracts on its own, by its own factor.

On a quadratic the coefficient along the eigenvector u_i evolves as
alpha_i^(T) = (1 - eta lambda_i)^T alpha_i^(0), so the directions never
interact. One learning rate has to serve all of them at once, and the
figure shows the consequence: the fastest-curving direction sets the
stability limit, the slowest-curving one sets the speed.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S

NAME = "eigen_decay"

LAMS = np.array([0.4, 1.5, 4.0, 9.0])
ETA = 0.20


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))
    T = np.arange(0, 61)
    cols = plt.get_cmap(S.CAT)(np.linspace(0.06, 0.78, len(LAMS)))

    ax = axes[0]
    for lam, col in zip(LAMS, cols):
        f = abs(1.0 - ETA * lam)
        ax.semilogy(T, np.maximum(f ** T, 1e-16), color=col, lw=2.4,
                    label=r"$\lambda = %.1f$,  $|1-\eta\lambda| = %.2f$"
                          % (lam, f))
    ax.set_xlabel("iteration $T$")
    ax.set_ylabel(r"$|\alpha_i^{(T)} / \alpha_i^{(0)}|$")
    ax.set_ylim(1e-8, 2)
    ax.legend(loc="upper right", fontsize=12)
    ax.set_title(r"(a) one $\eta = %.2f$, four curvatures" % ETA)
    S.square(ax)

    ax = axes[1]
    lam = np.linspace(0.0, 12.0, 500)
    for e, col in zip((0.08, 0.20, 0.24), (S.L2, S.L1, S.OUTC)):
        ax.plot(lam, np.abs(1.0 - e * lam), color=col, lw=2.4,
                label=r"$\eta = %.2f$" % e)
    ax.axhline(1.0, color=S.GREY, lw=1.4, ls=(0, (5, 3)))
    ax.annotate("divergence above this line", xy=(6.4, 1.06),
                fontsize=13.5, color=S.GREY, ha="center")
    ax.set_xlabel(r"curvature $\lambda$")
    ax.set_ylabel(r"contraction factor $|1 - \eta\lambda|$")
    ax.set_ylim(0, 1.8)
    ax.legend(loc="upper left", fontsize=13)
    ax.set_title(r"(b) stability needs $\eta < 2/\lambda$ for every $\lambda$")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

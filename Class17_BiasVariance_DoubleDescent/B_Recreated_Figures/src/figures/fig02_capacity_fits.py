"""What capacity trades: the same ensemble at two capacities.

Following the construction of Bishop & Bishop (2024), Fig. 4.7 and
Prince (2023), Fig. 8.8: L independent data sets, one fit per data set,
the ensemble mean compared with h(x).

(a) K = 4. The twenty fits lie almost on top of one another -- small
    variance -- but the mean departs from h(x) everywhere: large bias.

(b) K = 20. The twenty fits scatter widely -- large variance -- yet
    their mean tracks h(x) closely: small bias.

The measured values of the two terms are printed on each panel, so the
trade-off is a number rather than an impression.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import truth, sample, piecewise_fit

NAME = "capacity_fits"

GRID = np.linspace(0.02, 0.98, 400)
N = 60
L = 20
KS = (4, 20)


def _ensemble(K, seed):
    rng = np.random.default_rng(seed)
    F = np.empty((L, len(GRID)))
    for l in range(L):
        x, y = sample(N, rng)
        F[l] = piecewise_fit(x, y, K, grid=GRID)
    return F


def build():
    S.use()
    h = truth(GRID)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9))

    for ax, K, tag in zip(axes, KS, "ab"):
        F = _ensemble(K, seed=11)
        for l in range(L):
            ax.plot(GRID, F[l], color=S.L2, lw=0.9, alpha=0.40, zorder=3)
        fbar = F.mean(axis=0)
        ax.plot(GRID, fbar, color=S.L2, lw=2.9, zorder=5,
                label="ensemble mean")
        ax.plot(GRID, h, color=S.GREY, lw=2.6, zorder=4, label=r"$h(x)$")

        b2 = np.mean((fbar - h) ** 2)
        v = np.mean(F.var(axis=0))
        ax.annotate(r"bias$^2 = %.3f$" "\n" r"variance $= %.3f$" % (b2, v),
                    xy=(0.03, 0.035), xycoords="axes fraction",
                    ha="left", va="bottom", fontsize=14, color=S.OUTC)

        ax.set_xlim(0.02, 0.98)
        ax.set_ylim(-2.6, 2.3)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.legend(loc="upper right", fontsize=13)
        ax.set_title(r"(%s) $K = %d$" % (tag, K))
        ax.set_box_aspect(0.80)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

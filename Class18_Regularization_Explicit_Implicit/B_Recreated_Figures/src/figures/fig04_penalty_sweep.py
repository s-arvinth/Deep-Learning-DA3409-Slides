"""What the penalty coefficient buys, and what it costs.

Following the sweep of Prince (2023), Fig. 9.2: the same over-capacity
model fitted to the same noisy sample at three penalty strengths.

(a) The penalty is negligible and the fit chases every point.
(b) The penalty is doing its job; the fit is close to h(x).
(c) The penalty dominates and the fit is smoother than the truth, so it
    is now worse for the opposite reason.

The number printed on each panel is the mean squared error against the
true function over the interval, measured, not asserted.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import truth, sample, ridge_fit

NAME = "penalty_sweep"

GRID = np.linspace(0.0, 1.0, 600)
N, K = 22, 45
LAMS = ((1e-6, r"10^{-6}"), (3e-2, r"0.03"), (3.0, r"3"))


def build():
    S.use()
    rng = np.random.default_rng(6)
    x, y = sample(N, rng)
    h = truth(GRID)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))

    for ax, (lam, lab), tag in zip(axes, LAMS, "abc"):
        f, w = ridge_fit(x, y, K, lam, GRID)
        ax.plot(GRID, h, color=S.GREY, lw=2.4, zorder=3, label=r"$h(x)$")
        ax.plot(GRID, f, color=S.OUTC, lw=2.6, zorder=4, label="fit")
        ax.plot(x, y, "o", ms=6.5, mfc="white", mec=S.L1, mew=1.7, zorder=6)
        ax.annotate(r"$\|\mathbf{w}\| = %.1f$" % np.linalg.norm(w[1:]),
                    xy=(0.035, 0.94), xycoords="axes fraction", ha="left",
                    va="top", fontsize=13.5, color=S.L1)
        ax.annotate("error against $h$: %.3f" % np.mean((f - h) ** 2),
                    xy=(0.5, 0.045), xycoords="axes fraction", ha="center",
                    fontsize=13, color=S.GREY)
        ax.set_xlim(0, 1)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title(r"(%s) $\lambda = %s$" % (tag, lab))
        ax.set_box_aspect(0.78)

    axes[0].legend(loc="lower left", fontsize=12.5)
    fig.tight_layout(w_pad=1.5)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

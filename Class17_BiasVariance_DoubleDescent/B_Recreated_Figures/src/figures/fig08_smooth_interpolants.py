"""Why the over-parameterised regime is not absurd.

Following the argument of Prince (2023), Section 8.4.1 and Fig. 8.11.

Nine points, with a deliberate gap in the middle, are fitted exactly in
a basis of K bumps of FIXED width by minimum-norm least squares. Adding
bumps therefore adds candidate solutions without changing what a single
bump can do -- the same thing extra hidden units do.

Every curve shown has a training residual below 1e-14, so the training
loss cannot tell them apart. What changes is which interpolant gets
chosen:

(a) K = 9. There is exactly one interpolant, and it contorts itself to
    reach the points. ||w|| = 13.
(b) K = 11. A small subspace of interpolants; the smallest is already
    far tamer. ||w|| = 2.2.
(c) K = 400. ||w|| = 0.32, and the curve is smooth across the gap.

Extra capacity does not buy a better fit to the training data; there is
nothing left to fit. It buys a better choice among the fits that already
exist, and the norm falling from 13 to 0.32 is the same quantity that
peaks at the interpolation threshold in fig05.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import truth

NAME = "smooth_interpolants"

GRID = np.linspace(0.0, 1.0, 700)
XS = np.array([0.03, 0.10, 0.17, 0.25, 0.33, 0.76, 0.84, 0.91, 0.98])
KS = (9, 11, 400)
WIDTH = 0.07                       # fixed: adding bumps adds choices only
GAP = (0.36, 0.74)


def _bumps(x, K, s=WIDTH):
    c = np.linspace(-0.05, 1.05, K)
    return np.exp(-0.5 * ((x[:, None] - c[None, :]) / s) ** 2)


def build():
    S.use()
    ys = truth(XS)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))

    for ax, K, tag in zip(axes, KS, "abc"):
        Phi = _bumps(XS, K)
        w = np.linalg.pinv(Phi) @ ys            # minimum-norm interpolant
        f = _bumps(GRID, K) @ w
        resid = np.max(np.abs(Phi @ w - ys))

        ax.axvspan(*GAP, color=S.GREY, alpha=0.11, lw=0)
        ax.plot(GRID, truth(GRID), color=S.GREY, lw=2.3, zorder=3,
                label=r"$h(x)$")
        ax.plot(GRID, f, color=S.OUTC, lw=2.6, zorder=4,
                label="interpolant")
        ax.plot(XS, ys, "o", ms=8, mfc="white", mec=S.L1, mew=2.0, zorder=6)

        ax.annotate("no data here", xy=(np.mean(GAP), 0.945),
                    xycoords=("data", "axes fraction"), ha="center",
                    va="top", fontsize=12.5, color=S.GREY)
        ax.annotate(r"$\|\hat{\mathbf{w}}\| = %.2f$" % np.linalg.norm(w),
                    xy=(0.035, 0.94), xycoords="axes fraction", ha="left",
                    va="top", fontsize=14.5, color=S.OUTC)
        top = float(np.max(f))
        if top > 3.4:
            ax.annotate("curve continues\nup to %.1f" % top,
                        xy=(GRID[int(np.argmax(f))], 3.3),
                        xytext=(0.62, 0.62), textcoords="axes fraction",
                        fontsize=12, color=S.OUTC, ha="center",
                        arrowprops=dict(arrowstyle="-|>", color=S.OUTC,
                                        lw=1.3))
        ax.annotate("residual $< 10^{%d}$"
                    % int(np.ceil(np.log10(max(resid, 1e-16)))),
                    xy=(0.965, 0.055), xycoords="axes fraction", ha="right",
                    fontsize=12, color=S.GREY)

        ax.set_xlim(0, 1)
        ax.set_ylim(-2.4, 3.4)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title(r"(%s) $K = %d$" % (tag, K))
        ax.set_box_aspect(0.78)

    axes[2].legend(loc="upper right", fontsize=12.5)
    fig.tight_layout(w_pad=1.5)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

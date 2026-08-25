"""Five update rules on one ill-conditioned quadratic, by family.

Same starting point, same budget of iterations, each method at constants
that suit it. The two families are drawn on separate copies of the
surface -- the momentum family in (a), the adaptive family in (b) --
because putting five trajectories on one contour hides the shape of
every one of them. Panel (c) then compares all five by error, which is
the only place a single axis can carry them all.

The point is not which rule wins on this bowl. It is that the families
fail and succeed for different reasons: momentum attacks the ratio
lambda_max / lambda_min, while the adaptive rules attack the scale of
each coordinate separately.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import (quad_E, quad_grad, gd, momentum, nesterov,
                           rmsprop, adam, gd_best, heavyball_best,
                           nesterov_best)

NAME = "optimiser_compare"

LAM = (0.50, 6.00)
START = np.array([-2.40, 0.90])
N = 60

XLIM, YLIM = 2.80, 1.25


def _runs():
    G = lambda w: quad_grad(w, lam=LAM)
    e0, _ = gd_best(LAM)
    e1, m1, _ = heavyball_best(LAM)
    e2, m2 = nesterov_best(LAM)
    grp_a = [("gradient descent", gd(G, START, e0, N), S.GREY),
             ("momentum", momentum(G, START, e1, m1, N), S.OUTC),
             ("Nesterov", nesterov(G, START, e2, m2, N), S.L2)]
    grp_b = [("RMSProp", rmsprop(G, START, 0.10, 0.9, N), S.ACC),
             ("Adam", adam(G, START, 0.14, 0.9, 0.99, N), S.L1)]
    return grp_a, grp_b


def _contour_panel(ax, runs, title):
    g1 = np.linspace(-XLIM, XLIM, 340)
    g2 = np.linspace(-YLIM, YLIM, 220)
    G1, G2 = np.meshgrid(g1, g2)
    Z = quad_E(G1, G2, lam=LAM)
    ax.contour(G1, G2, Z, levels=np.array([0.35, 0.9, 1.8, 3.0]),
               colors=[S.GREY], linewidths=0.9, alpha=0.45)
    for lab, p, col in runs:
        q = p[:34]
        q = q[np.abs(q).max(axis=1) <= 3.2]
        ax.plot(q[:, 0], q[:, 1], "-", color=col, lw=2.1, label=lab,
                zorder=4)
    ax.plot(*START, "o", ms=9, mfc="white", mec=S.OUTC, mew=2.0, zorder=6)
    ax.plot(0, 0, "*", ms=15, mfc="white", mec=S.OUTC, mew=1.8, zorder=6)
    ax.set_xlim(-XLIM, XLIM)
    ax.set_ylim(-YLIM, YLIM)
    ax.set_xlabel("$w_1$")
    ax.set_ylabel("$w_2$")
    ax.legend(loc="upper right", fontsize=11.5, ncol=1,
              handlelength=1.4, borderpad=0.35, labelspacing=0.25)
    ax.set_title(title)
    ax.set_box_aspect(0.62)


def build():
    S.use()
    grp_a, grp_b = _runs()
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.6))

    _contour_panel(axes[0], grp_a, "(a) the momentum family")
    _contour_panel(axes[1], grp_b, "(b) the adaptive family")

    ax = axes[2]
    for lab, p, col in grp_a + grp_b:
        E = quad_E(p[:, 0], p[:, 1], lam=LAM)
        ax.semilogy(np.maximum(E, 1e-16), color=col, lw=2.0, label=lab)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$E(\mathbf{w})$")
    ax.set_xlim(0, N)
    ax.set_ylim(1e-10, 1e2)
    ax.legend(loc="upper right", fontsize=11.5, handlelength=1.4,
              borderpad=0.35, labelspacing=0.25)
    ax.set_title("(c) all five, by error")
    ax.set_box_aspect(0.62)

    fig.tight_layout(w_pad=1.5)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

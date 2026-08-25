"""Five update rules on one ill-conditioned quadratic, by family.

Same starting point, same budget of iterations, each method at constants
that suit it. Two figures, so that neither is crowded and neither is
wider than a slide:

    optimiser_compare  (a) the momentum family   (b) the adaptive family
    optimiser_cost     (a) all five by error     (b) iterations to 1e-6

Putting five trajectories on one contour hides the shape of every one of
them, so the families get separate panels; the error axis is the only
one that can carry all five at once.

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
TOL = 1e-6


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
               colors=[S.GREY], linewidths=1.0, alpha=0.45)
    for lab, p, col in runs:
        q = p[:34]
        q = q[np.abs(q).max(axis=1) <= 3.2]
        ax.plot(q[:, 0], q[:, 1], "-", color=col, lw=2.3, label=lab,
                zorder=4)
    ax.plot(*START, "o", ms=10, mfc="white", mec=S.OUTC, mew=2.1, zorder=6)
    ax.plot(0, 0, "*", ms=17, mfc="white", mec=S.OUTC, mew=1.9, zorder=6)
    ax.set_xlim(-XLIM, XLIM)
    ax.set_ylim(-YLIM, YLIM)
    ax.set_xlabel("$w_1$")
    ax.set_ylabel("$w_2$")
    ax.legend(loc="upper right", fontsize=13, handlelength=1.4,
              borderpad=0.35, labelspacing=0.28)
    ax.set_title(title)
    ax.set_box_aspect(0.55)


def _families():
    grp_a, grp_b = _runs()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    _contour_panel(axes[0], grp_a, "(a) the momentum family")
    _contour_panel(axes[1], grp_b, "(b) the adaptive family")
    fig.tight_layout(w_pad=1.8)
    S.save(fig, "optimiser_compare")


def _cost():
    grp_a, grp_b = _runs()
    allr = grp_a + grp_b
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0))

    ax = axes[0]
    for lab, p, col in allr:
        E = quad_E(p[:, 0], p[:, 1], lam=LAM)
        ax.semilogy(np.maximum(E, 1e-16), color=col, lw=2.2, label=lab)
    ax.axhline(TOL, color=S.GREY, lw=1.1, ls=(0, (4, 3)))
    ax.annotate(r"$E = 10^{-6}$", xy=(1.5, TOL * 3.0), color=S.GREY,
                fontsize=13, ha="left")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$E(\mathbf{w})$")
    ax.set_xlim(0, N)
    ax.set_ylim(1e-10, 1e2)
    ax.legend(loc="upper right", fontsize=12.5, handlelength=1.4,
              borderpad=0.35, labelspacing=0.28)
    ax.set_title("(a) all five, by error")
    S.square(ax)

    ax = axes[1]
    labs, vals, cols, hits = [], [], [], []
    for lab, p, col in allr:
        E = quad_E(p[:, 0], p[:, 1], lam=LAM)
        hit = np.flatnonzero(E < TOL)
        labs.append(lab)
        vals.append(int(hit[0]) if hit.size else N)
        hits.append(bool(hit.size))
        cols.append(col)
    y = np.arange(len(labs))[::-1]
    for yy, v, c, ok in zip(y, vals, cols, hits):
        ax.barh(yy, v, color=c, height=0.58,
                alpha=1.0 if ok else 0.30,
                hatch=None if ok else "//",
                edgecolor="none" if ok else c)
        if ok:
            ax.annotate("%d" % v, xy=(v + 1.0, yy), va="center",
                        fontsize=13.5, color=S.GREY)
        else:
            ax.annotate("not reached in %d" % N, xy=(v - 1.5, yy),
                        va="center", ha="right", fontsize=13, color=c)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=14)
    ax.set_xlim(0, N)
    ax.set_xlabel(r"iterations to reach $E < 10^{-6}$")
    ax.set_title("(b) the same runs, as a count")
    ax.set_box_aspect(0.86)

    fig.tight_layout(w_pad=2.0)
    S.save(fig, "optimiser_cost")


def build():
    S.use()
    _families()
    _cost()


if __name__ == "__main__":
    build()

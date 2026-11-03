"""optimiser_compare — five update rules on one ill-conditioned quadratic, by family

Used in: Class 14 (B), slide(s): "Five update rules, by family"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python optimiser_compare.py            -> output/optimiser_compare.pdf
         python optimiser_compare.py --out DIR  -> DIR/optimiser_compare.pdf
Ported from: Class14_Convergence/B_Recreated_Figures/src/figures/fig11_optimiser_compare.py

Five update rules on one ill-conditioned quadratic, by family.

Same starting point, same budget of iterations, each method at constants
that suit it. Two figures, so that neither is crowded and neither is
wider than a slide:

    optimiser_compare  (a) the momentum family   (b) the adaptive family

(The original module also wrote optimiser_cost.pdf, which no deck loads;
that part is dropped here.)

Putting five trajectories on one contour hides the shape of every one of
them, so the families get separate panels; the error axis is the only
one that can carry all five at once.

The point is not which rule wins on this bowl. It is that the families
fail and succeed for different reasons: momentum attacks the ratio
lambda_max / lambda_min, while the adaptive rules attack the scale of
each coordinate separately.
"""
import numpy as np
import matplotlib.pyplot as plt
import common as S
from models import (quad_E, quad_grad, gd, momentum, nesterov,
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
    S.save(fig, NAME)


def build():
    S.use()
    _families()


if __name__ == "__main__":
    build()

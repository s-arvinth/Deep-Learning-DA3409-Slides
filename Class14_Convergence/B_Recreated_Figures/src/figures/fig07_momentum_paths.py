"""Gradient descent, momentum and Nesterov momentum, one panel each.

Same valley, same start, same number of steps, each method run with the
constants that are optimal for this quadratic. Drawing the three
trajectories on separate copies of the surface keeps every path
readable: the shape of each one is the point, and overlaying them hides
exactly the shape we are trying to see.

Read left to right: the stride along the valley lengthens and the
crossing of the valley disappears.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import (quad_E, quad_grad, gd, momentum, nesterov,
                           gd_best, heavyball_best, nesterov_best)

NAME = "momentum_paths"

LAM = (0.60, 6.00)
START = np.array([-2.30, 0.88])
N = 45

XLIM, YLIM = 2.80, 1.25


def runs():
    G = lambda w: quad_grad(w, lam=LAM)
    e0, r0 = gd_best(LAM)
    e1, m1, r1 = heavyball_best(LAM)
    e2, m2 = nesterov_best(LAM)
    return [("(a) gradient descent", r"$\rho = %.3f$" % r0,
             gd(G, START, e0, N), S.GREY),
            ("(b) momentum", r"$\mu = %.2f$,  $\rho = %.3f$" % (m1, r1),
             momentum(G, START, e1, m1, N), S.OUTC),
            ("(c) Nesterov momentum", r"$\mu = %.2f$" % m2,
             nesterov(G, START, e2, m2, N), S.L2)]


def build():
    S.use()
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.4))

    g1 = np.linspace(-XLIM, XLIM, 340)
    g2 = np.linspace(-YLIM, YLIM, 220)
    G1, G2 = np.meshgrid(g1, g2)
    Z = quad_E(G1, G2, lam=LAM)

    for ax, (title, sub, p, col) in zip(axes, runs()):
        ax.contour(G1, G2, Z, levels=np.array([0.35, 0.9, 1.8, 3.0]),
                   colors=[S.GREY], linewidths=0.9, alpha=0.5)
        ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=2.2, zorder=4)
        ax.plot(p[:20, 0], p[:20, 1], "o", ms=3.4, mfc=col, mec="none",
                zorder=5)
        ax.plot(*START, "o", ms=9, mfc="white", mec=S.OUTC, mew=2.0,
                zorder=6)
        ax.plot(0, 0, "*", ms=15, mfc="white", mec=S.OUTC, mew=1.8,
                zorder=6)
        ax.annotate(sub, xy=(0.5, 0.045), xycoords="axes fraction",
                    color=col, fontsize=13.5, ha="center")
        ax.set_xlim(-XLIM, XLIM)
        ax.set_ylim(-YLIM, YLIM)
        ax.set_xlabel("$w_1$")
        ax.set_ylabel("$w_2$")
        ax.set_title(title)
        ax.set_box_aspect(0.52)

    fig.tight_layout(w_pad=1.5)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

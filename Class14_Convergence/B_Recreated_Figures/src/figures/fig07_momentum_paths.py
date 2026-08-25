"""Gradient descent, momentum and Nesterov momentum, two panels at a time.

Same valley, same start, same number of steps, each method run with the
constants that are optimal for this quadratic. Drawing at most two
trajectories per figure -- on separate copies of the surface -- keeps
every path readable and keeps the figure inside the slide.

    momentum_paths   (a) gradient descent   (b) momentum
    nesterov_path    (a) momentum           (b) Nesterov momentum

Momentum is repeated as the reference in the second figure, so each
comparison is against the picture the reader has just seen.
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
    """The three trajectories, keyed by short name."""
    G = lambda w: quad_grad(w, lam=LAM)
    e0, r0 = gd_best(LAM)
    e1, m1, r1 = heavyball_best(LAM)
    e2, m2 = nesterov_best(LAM)
    return {
        "gd": ("gradient descent", r"$\rho = %.3f$" % r0,
               gd(G, START, e0, N), S.GREY),
        "mom": ("momentum", r"$\mu = %.2f$,  $\rho = %.3f$" % (m1, r1),
                momentum(G, START, e1, m1, N), S.OUTC),
        "nes": ("Nesterov momentum", r"$\mu = %.2f$" % m2,
                nesterov(G, START, e2, m2, N), S.L2),
    }


def _panel(ax, entry, tag, mesh):
    G1, G2, Z = mesh
    title, sub, p, col = entry
    ax.contour(G1, G2, Z, levels=np.array([0.35, 0.9, 1.8, 3.0]),
               colors=[S.GREY], linewidths=1.0, alpha=0.5)
    ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=2.4, zorder=4)
    ax.plot(p[:20, 0], p[:20, 1], "o", ms=4.2, mfc=col, mec="none", zorder=5)
    ax.plot(*START, "o", ms=10, mfc="white", mec=S.OUTC, mew=2.1, zorder=6)
    ax.plot(0, 0, "*", ms=17, mfc="white", mec=S.OUTC, mew=1.9, zorder=6)
    ax.annotate(sub, xy=(0.5, 0.045), xycoords="axes fraction", color=col,
                fontsize=15, ha="center")
    ax.set_xlim(-XLIM, XLIM)
    ax.set_ylim(-YLIM, YLIM)
    ax.set_xlabel("$w_1$")
    ax.set_ylabel("$w_2$")
    ax.set_title("(%s) %s" % (tag, title))
    ax.set_box_aspect(0.55)


def _mesh():
    g1 = np.linspace(-XLIM, XLIM, 340)
    g2 = np.linspace(-YLIM, YLIM, 220)
    G1, G2 = np.meshgrid(g1, g2)
    return G1, G2, quad_E(G1, G2, lam=LAM)


def _pair(keys, name):
    R, mesh = runs(), _mesh()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    for ax, k, tag in zip(axes, keys, "ab"):
        _panel(ax, R[k], tag, mesh)
    fig.tight_layout(w_pad=1.8)
    S.save(fig, name)


def build():
    S.use()
    _pair(("gd", "mom"), "momentum_paths")
    _pair(("mom", "nes"), "nesterov_path")


if __name__ == "__main__":
    build()

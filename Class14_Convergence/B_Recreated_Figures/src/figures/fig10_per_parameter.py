"""Why one learning rate cannot serve two very different curvatures.

(a) an eta that makes good progress across the valley leaves the long
direction barely moving. (b) an eta chosen for the long direction is
unstable across the valley. (c) normalising the gradient to its sign
moves a fixed distance along each axis and makes progress in both, but
cannot settle. (d) Adam smooths both the gradient and the normaliser and
does settle.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import quad_E, quad_grad, gd, signsgd, adam

NAME = "per_parameter"

LAM = (0.30, 12.0)
START = np.array([-2.4, 0.85])


def build():
    S.use()
    G = lambda w: quad_grad(w, lam=LAM)
    cases = [
        (r"(a) $\eta$ suits the steep axis", gd(G, START, 0.14, 70), S.OUTC),
        (r"(b) $\eta$ suits the flat axis", gd(G, START, 0.175, 24), S.ACC),
        (r"(c) sign only", signsgd(G, START, 0.09, 70), S.L2),
        (r"(d) Adam", adam(G, START, 0.09, 0.9, 0.99, 70), S.L1),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16.4, 4.6))
    g1 = np.linspace(-2.9, 2.9, 320)
    g2 = np.linspace(-1.15, 1.15, 200)
    G1, G2 = np.meshgrid(g1, g2)
    Z = quad_E(G1, G2, lam=LAM)
    for ax, (title, p, col) in zip(axes, cases):
        ax.contour(G1, G2, Z, levels=np.array([0.2, 0.6, 1.3, 2.3]),
                   colors=[S.GREY], linewidths=0.9, alpha=0.55)
        q = p[np.abs(p).max(axis=1) <= 3.2]
        ax.plot(q[:, 0], q[:, 1], "-o", color=col, lw=1.6, ms=3.6,
                mfc="white", mew=0.9, zorder=4)
        ax.plot(*START, "o", ms=8, mfc=col, mec="white", mew=1.3, zorder=6)
        ax.plot(0, 0, "*", ms=14, mfc="white", mec=S.OUTC, mew=1.7,
                zorder=6)
        ax.set_xlim(-2.9, 2.9); ax.set_ylim(-1.15, 1.15)
        ax.set_xlabel("$w_1$")
        ax.set_title(title, fontsize=15.5)
        ax.set_box_aspect(0.60)
    axes[0].set_ylabel("$w_2$")
    fig.tight_layout(w_pad=1.0)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

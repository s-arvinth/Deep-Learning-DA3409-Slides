"""Why the step goes along the negative gradient, and nowhere else.

Bishop & Bishop (2024), Eq. 7.1: to first order a step d changes the
error by d^T grad E.  Over unit-length directions that inner product is
minimised at d = -grad E / ||grad E||, by Cauchy-Schwarz.

Panel (a) plots the predicted change against the angle between the step
and the negative gradient -- a cosine, minimised at zero.  Panel (b) is
the same statement drawn on the contour map: only directions inside the
half-plane where the cosine is negative go downhill at all.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import quadratic_E, quadratic_grad

NAME = "steepest_direction"


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2))

    # ---- (a) the cosine ----------------------------------------------
    ax = axes[0]
    th = np.linspace(-np.pi, np.pi, 400)
    ax.plot(np.rad2deg(th), -np.cos(th), color=S.L1, lw=2.4)
    ax.axhline(0.0, color=S.GREY, lw=0.9, ls=(0, (4, 3)))
    ax.plot(0, -1, "o", ms=10, mfc="white", mec=S.OUTC, mew=2.2, zorder=5)
    ax.annotate("steepest descent", xy=(0, -1), xytext=(6, -0.62),
                fontsize=15, color=S.OUTC,
                arrowprops=dict(arrowstyle="-|>", color=S.OUTC, lw=1.4))
    ax.fill_between(np.rad2deg(th), -np.cos(th), 0.0,
                    where=(-np.cos(th) < 0), color=S.L2, alpha=0.16)
    ax.annotate("downhill", xy=(-62, -0.35), fontsize=15, color=S.L2)
    ax.annotate("uphill", xy=(118, 0.42), fontsize=15, color=S.GREY)
    ax.set_xlabel(r"angle to $-\nabla E$  (degrees)")
    ax.set_ylabel(r"$\Delta E \,/\, \|\Delta w\|\,\|\nabla E\|$")
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_title("(a) predicted change, by direction")
    S.square(ax)

    # ---- (b) the same thing on the map -------------------------------
    ax = axes[1]
    g = np.linspace(-2.4, 2.4, 300)
    W1, W2 = np.meshgrid(g, g)
    Z = quadratic_E(W1, W2, lam=(1.0, 3.0), rot=np.deg2rad(28))
    ax.contour(W1, W2, Z, levels=8, colors=[S.GREY], linewidths=0.8,
               alpha=0.65)

    p = np.array([-1.55, 1.15])
    gr = quadratic_grad(p, lam=(1.0, 3.0), rot=np.deg2rad(28))
    u = -gr / np.linalg.norm(gr)
    for ang, col, lw, lab in ((0.0, S.OUTC, 2.6, r"$-\nabla E$"),
                              (55.0, S.L2, 1.8, None),
                              (-55.0, S.L2, 1.8, None),
                              (118.0, S.GREY, 1.4, None)):
        a = np.deg2rad(ang)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        d = R @ u * 1.15
        ax.annotate("", xy=tuple(p + d), xytext=tuple(p),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=lw))
        if lab:
            ax.annotate(lab, xy=tuple(p + 1.30 * d), color=col, fontsize=16,
                        ha="center", va="center")
    ax.plot(*p, "o", ms=8, mfc="white", mec=S.L1, mew=2.0, zorder=6)
    ax.plot(0, 0, "*", ms=15, mfc="white", mec=S.OUTC, mew=1.8, zorder=6)
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.4, 2.4)
    ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
    ax.set_title("(b) the same claim on the contours")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

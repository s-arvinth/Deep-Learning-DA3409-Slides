"""The four step-size regimes, walked out on the surface itself.

The same runs as the contour version, drawn on the error surface so the
altitude is visible: too small barely descends, 1/L drops fastest, the
intermediate value bounces across the valley on the way down, and above
2/L each step lands higher than the last.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import quadratic_E, quadratic_grad

NAME = "step_regimes_3d"

LAM = (0.6, 4.0)
START = np.array([-2.05, 1.85])
CASES = [(0.05, r"$\eta \ll 1/L$", S.L2, 60),
         (0.25, r"$\eta = 1/L$", S.L1, 40),
         (0.42, r"$1/L < \eta < 2/L$", S.ACC, 40),
         (0.54, r"$\eta > 2/L$", S.OUTC, 9)]


def build():
    S.use()
    g = np.linspace(-3.0, 3.0, 90)
    G1, G2 = np.meshgrid(g, g)
    Z = quadratic_E(G1, G2, lam=LAM)

    fig = plt.figure(figsize=(16.4, 4.7))
    for k, (eta, sym, col, n) in enumerate(CASES):
        ax = fig.add_subplot(1, 4, k + 1, projection="3d")
        ax.plot_surface(G1, G2, Z, cmap=S.SEQ, linewidth=0, alpha=0.55,
                        rstride=2, cstride=2, rasterized=True)
        w = START.copy(); p = [w.copy()]
        for _ in range(n):
            w = w - eta * quadratic_grad(w, lam=LAM)
            p.append(w.copy())
            if np.abs(w).max() > 3.0:
                break
        p = np.array(p)
        p = p[np.abs(p).max(axis=1) <= 3.0]
        E = quadratic_E(p[:, 0], p[:, 1], lam=LAM)
        ax.plot(p[:, 0], p[:, 1], E + 0.25, "-o", color=col, lw=2.0,
                ms=3.6, zorder=10)
        ax.plot([0], [0], [0.25], "*", ms=11, color=S.OUTC, zorder=11)
        ax.view_init(elev=34, azim=-62)
        ax.set_xlabel("$w_1$", labelpad=-8)
        ax.set_ylabel("$w_2$", labelpad=-8)
        S.tidy3d(ax, ticks=(-2, 0, 2), labelsize=10)
        ax.set_zticks([])
        ax.set_zlim(0, 12)
        ax.set_title(sym, fontsize=17, pad=-4)
    fig.tight_layout(w_pad=0.2)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

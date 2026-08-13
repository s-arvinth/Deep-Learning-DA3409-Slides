"""The four step-size regimes, on the contour map.

The same bowl and the same starting point in every panel; only eta
changes. With lambda_max = 4 the critical value is 2/L = 0.5.

    eta = 0.05  far below 1/L : every step helps, but barely
    eta = 0.25  equal to 1/L  : the largest guaranteed decrease
    eta = 0.42  between 1/L and 2/L : still downhill, now oscillating
    eta = 0.54  above 2/L     : the steps grow and the iterate leaves
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import quadratic_E, quadratic_grad

NAME = "step_regimes_2d"

LAM = (0.6, 4.0)
START = np.array([-2.05, 1.85])
CASES = [(0.05, r"$\eta \ll 1/L$", "crawls", S.L2, 60),
         (0.25, r"$\eta = 1/L$", "fastest guarantee", S.L1, 40),
         (0.42, r"$1/L < \eta < 2/L$", "oscillates, still falls", S.ACC, 40),
         (0.54, r"$\eta > 2/L$", "diverges", S.OUTC, 12)]


def build():
    S.use()
    g = np.linspace(-3.0, 3.0, 300)
    G1, G2 = np.meshgrid(g, g)
    Z = quadratic_E(G1, G2, lam=LAM)

    fig, axes = plt.subplots(1, 4, figsize=(16.4, 4.6))
    for ax, (eta, sym, word, col, n) in zip(axes, CASES):
        ax.contour(G1, G2, Z, levels=np.array([0.3, 1.0, 2.2, 4.0, 6.4]),
                   colors=[S.GREY], linewidths=0.8, alpha=0.55)
        w = START.copy()
        p = [w.copy()]
        for _ in range(n):
            w = w - eta * quadratic_grad(w, lam=LAM)
            if np.abs(w).max() > 8.0:
                p.append(w.copy()); break
            p.append(w.copy())
        p = np.array(p)
        ax.plot(p[:, 0], p[:, 1], "-o", color=col, lw=1.6, ms=4.2,
                mfc="white", mew=1.0, zorder=4)
        ax.plot(*START, "o", ms=8, mfc=col, mec="white", mew=1.4, zorder=6)
        ax.plot(0, 0, "*", ms=14, mfc="white", mec=S.OUTC, mew=1.7,
                zorder=6)
        ax.set_xlim(-3.0, 3.0); ax.set_ylim(-3.0, 3.0)
        ax.set_xlabel("$w_1$")
        ax.set_title("%s\n%s" % (sym, word), fontsize=16)
        S.square(ax)
    axes[0].set_ylabel("$w_2$")
    fig.tight_layout(w_pad=1.2)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

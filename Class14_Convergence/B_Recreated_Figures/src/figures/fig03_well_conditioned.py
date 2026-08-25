"""When every direction curves the same, one step is enough.

With lambda_1 = lambda_2 the contours are circles, the negative gradient
points straight at the minimum, and eta = 1/lambda lands on it exactly.
Nothing in later sections improves on this; the whole of the rest of the
class is about what to do when it fails.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import quad_E, quad_grad

NAME = "well_conditioned"

START = np.array([-2.0, 1.6])


def build():
    S.use()
    g = np.linspace(-2.8, 2.8, 300)
    G1, G2 = np.meshgrid(g, g)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0))
    cases = [((2.0, 2.0), 1.0 / 2.0, r"$\kappa = 1$,  $\eta = 1/\lambda$",
              "one step"),
             ((2.0, 2.0), 0.15, r"$\kappa = 1$,  $\eta$ too small",
              "many small steps"),
             ((2.0, 12.0), 1.0 / 12.0, r"$\kappa = 6$,  $\eta = 1/\lambda_{max}$",
              "safe, but slow")]

    for ax, (lam, eta, title, note) in zip(axes, cases):
        Z = quad_E(G1, G2, lam=lam)
        ax.contour(G1, G2, Z, levels=np.array([0.5, 2.0, 4.5, 8.0, 13.0]),
                   colors=[S.GREY], linewidths=0.9, alpha=0.6)
        w = START.copy(); p = [w.copy()]
        for _ in range(30):
            w = w - eta * quad_grad(w, lam=lam)
            p.append(w.copy())
        p = np.array(p)
        ax.plot(p[:, 0], p[:, 1], "-o", color=S.OUTC, lw=1.8, ms=5.0,
                mfc="white", mew=1.3, zorder=4)
        ax.plot(*START, "o", ms=9, mfc=S.OUTC, mec="white", mew=1.4,
                zorder=6)
        ax.plot(0, 0, "*", ms=15, mfc="white", mec=S.OUTC, mew=1.8,
                zorder=6)
        ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.8, 2.8)
        ax.set_xlabel("$w_1$")
        ax.set_title("%s\n%s" % (title, note), fontsize=15.5)
        S.square(ax)
    axes[0].set_ylabel("$w_2$")

    fig.tight_layout(w_pad=1.4)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

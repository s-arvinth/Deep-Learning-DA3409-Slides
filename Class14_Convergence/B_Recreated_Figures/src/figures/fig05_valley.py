"""The long valley, and why a fixed step size crawls along it.

The error surface has very different curvature along the two
eigen-directions. The negative gradient at almost every point is nearly
perpendicular to the valley floor rather than along it, so successive
steps oscillate across the valley and progress towards the minimum is
slow.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import quad_E, quad_grad

NAME = "valley"

LAM = (0.35, 9.0)
START = np.array([-2.35, 0.72])


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))
    g1 = np.linspace(-2.8, 2.8, 340)
    g2 = np.linspace(-1.1, 1.1, 200)
    G1, G2 = np.meshgrid(g1, g2)
    Z = quad_E(G1, G2, lam=LAM)

    ax = axes[0]
    ax.contour(G1, G2, Z, levels=np.array([0.25, 0.7, 1.4, 2.4]),
               colors=[S.GREY], linewidths=1.0, alpha=0.7)
    eta = 0.21
    w = START.copy(); p = [w.copy()]
    for _ in range(40):
        w = w - eta * quad_grad(w, lam=LAM)
        p.append(w.copy())
    p = np.array(p)
    ax.plot(p[:, 0], p[:, 1], "-o", color=S.OUTC, lw=1.6, ms=4.4,
            mfc="white", mew=1.1, zorder=4)
    ax.plot(0, 0, "*", ms=15, mfc="white", mec=S.OUTC, mew=1.8, zorder=6)
    ax.annotate(r"$\mathbf{u}_1$", xy=(2.35, 0.10), fontsize=16, color=S.L1)
    ax.annotate(r"$\mathbf{u}_2$", xy=(0.12, 0.88), fontsize=16, color=S.L1)
    ax.annotate("", xy=(2.2, 0.0), xytext=(-2.2, 0.0),
                arrowprops=dict(arrowstyle="<|-|>", color=S.L1, lw=1.4))
    ax.annotate("", xy=(0.0, 0.82), xytext=(0.0, -0.82),
                arrowprops=dict(arrowstyle="<|-|>", color=S.L1, lw=1.4))
    ax.set_xlim(-2.8, 2.8); ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
    ax.set_title(r"(a) $\kappa = %.0f$: across fast, along slowly"
                 % (LAM[1] / LAM[0]))
    ax.set_box_aspect(0.42)

    ax = axes[1]
    for e, col, lab in ((0.05, S.L2, r"$\eta = 0.05$"),
                        (0.21, S.L1, r"$\eta = 0.21$"),
                        (0.222, S.OUTC, r"$\eta = 0.222$")):
        w = START.copy(); E = []
        for _ in range(200):
            E.append(quad_E(w[0], w[1], lam=LAM))
            w = w - e * quad_grad(w, lam=LAM)
            if not np.isfinite(w).all() or np.abs(w).max() > 1e6:
                break
        ax.semilogy(np.maximum(E, 1e-14), color=col, lw=2.2, label=lab)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$E(\mathbf{w})$")
    ax.set_ylim(1e-6, 1e2)
    ax.legend(loc="upper right", fontsize=13)
    ax.set_title(r"(b) even the best fixed $\eta$ is slow")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

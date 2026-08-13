"""Convex and non-convex, side by side, with the chord test.

Panel (a): a convex function. Every chord lies on or above the graph,
there is one stationary point, and it is the global minimum.

Panel (b): the same picture for a non-convex function. One chord cuts
below the graph, and the function has several stationary points of
three different kinds.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S

NAME = "convex_vs_nonconvex"


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2))

    x = np.linspace(-2.6, 2.6, 500)

    # ---- (a) convex ---------------------------------------------------
    ax = axes[0]
    f = 0.42 * x ** 2 + 0.35
    ax.plot(x, f, color=S.L1, lw=2.6, zorder=3)
    for a, b, col in ((-2.2, 1.5, S.OUTC), (-0.6, 2.3, S.ACC)):
        fa, fb = 0.42 * a ** 2 + 0.35, 0.42 * b ** 2 + 0.35
        ax.plot([a, b], [fa, fb], color=col, lw=1.8, ls=(0, (5, 3)),
                zorder=4)
        ax.plot([a, b], [fa, fb], "o", ms=6.5, color=col, zorder=5)
    ax.plot(0, 0.35, "*", ms=17, mfc="white", mec=S.OUTC, mew=1.8,
            zorder=6)
    ax.annotate("the only stationary point,\nand it is global",
                xy=(0.02, 0.32), xytext=(1.15, 2.70), fontsize=13.5,
                ha="center", color=S.OUTC,
                arrowprops=dict(arrowstyle="-|>", color=S.OUTC, lw=1.3))
    ax.set_xlabel("$w$"); ax.set_ylabel("$E(w)$")
    ax.set_ylim(-0.15, 3.5)
    ax.set_title("(a) convex: every chord sits above")
    S.square(ax)

    # ---- (b) non-convex -----------------------------------------------
    ax = axes[1]
    g = (0.30 * x ** 2 + 0.9 * np.sin(2.6 * x) * np.exp(-0.18 * x ** 2)
         + 1.05)
    ax.plot(x, g, color=S.L1, lw=2.6, zorder=3)

    def gv(t):
        return (0.30 * t ** 2 + 0.9 * np.sin(2.6 * t)
                * np.exp(-0.18 * t ** 2) + 1.05)

    a, b = -1.9, 1.35
    ax.plot([a, b], [gv(a), gv(b)], color=S.OUTC, lw=1.8, ls=(0, (5, 3)),
            zorder=4)
    ax.plot([a, b], [gv(a), gv(b)], "o", ms=6.5, color=S.OUTC, zorder=5)
    tt = np.linspace(a, b, 300)
    chord = gv(a) + (gv(b) - gv(a)) * (tt - a) / (b - a)
    ax.fill_between(tt, chord, gv(tt), where=(gv(tt) > chord),
                    color=S.OUTC, alpha=0.16, zorder=2)
    ax.annotate("this chord cuts below", xy=(0.05, chord.max() * 0.55),
                xytext=(-1.15, 2.85), fontsize=13.5, ha="center",
                color=S.OUTC,
                arrowprops=dict(arrowstyle="-|>", color=S.OUTC, lw=1.3))

    # mark the stationary points of g
    dg = np.gradient(g, x)
    sgn = np.sign(dg)
    idx = np.where(np.diff(sgn) != 0)[0]
    for i in idx:
        ax.plot(x[i], g[i], "o", ms=8, mfc="white", mec=S.L2, mew=2.0,
                zorder=6)
    j = int(np.argmin(g))
    ax.plot(x[j], g[j], "*", ms=17, mfc="white", mec=S.OUTC, mew=1.8,
            zorder=7)
    ax.plot([], [], "o", ms=7, mfc="white", mec=S.L2, mew=1.8,
            label="stationary")
    ax.plot([], [], "*", ms=13, mfc="white", mec=S.OUTC, mew=1.6,
            label="global minimum")
    ax.legend(loc="upper center", fontsize=13)
    ax.set_xlabel("$w$"); ax.set_ylabel("$E(w)$")
    ax.set_ylim(-0.15, 3.5)
    ax.set_title("(b) non-convex: one chord is enough to break it")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

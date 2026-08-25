"""How far is safe: the quadratic upper bound on one step.

For an L-smooth error function the descent lemma bounds the error after
a step of size eta along the negative gradient by a parabola in eta:

    E(w - eta grad E) <= E(w) - eta(1 - L eta / 2) ||grad E||^2 .

Panel (a) draws the true error along that line together with the bound,
marks the bound's own minimiser at eta = 1/L, and marks eta = 2/L where
the guarantee expires.  Panel (b) is the familiar step-size picture on a
one-dimensional quadratic: too small crawls, right lands, too big
overshoots and diverges.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S

NAME = "descent_lemma"


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))

    # ---- (a) the bound in eta ----------------------------------------
    ax = axes[0]
    L, g2 = 4.0, 1.0
    eta = np.linspace(0.0, 0.62, 400)
    bound = -eta * (1.0 - L * eta / 2.0) * g2
    # a true slice whose curvature along the line is 0.55 L, so it is
    # L-smooth and therefore has to lie on or below the bound
    true = -eta * (1.0 - 0.55 * L * eta / 2.0) * g2

    ax.axhline(0.0, color=S.GREY, lw=0.9, ls=(0, (4, 3)))
    ax.plot(eta, bound, color=S.L1, lw=2.4,
            label=r"guaranteed bound")
    ax.plot(eta, true, color=S.OUTC, lw=2.4, label=r"a true slice")
    ax.plot(1.0 / L, -g2 / (2 * L), "o", ms=10, mfc="white",
            mec=S.L1, mew=2.2, zorder=5)
    ax.annotate(r"$\eta = 1/L$", xy=(1 / L, -g2 / (2 * L)),
                xytext=(1 / L + 0.03, -g2 / (2 * L) - 0.055), fontsize=15,
                color=S.L1)
    ax.axvline(2.0 / L, color=S.ACC, lw=1.6, ls=(0, (5, 3)))
    ax.annotate(r"$\eta = 2/L$", xy=(2 / L - 0.012, -0.028), fontsize=15,
                color=S.ACC, ha="right", va="center")
    ax.fill_between(eta, bound, 0.0, where=(bound < 0),
                    color=S.L1, alpha=0.12)
    ax.set_xlabel(r"step size  $\eta$")
    ax.set_ylabel(r"$E(\mathbf{w} - \eta \nabla E) - E(\mathbf{w})$")
    ax.set_xlim(0, 0.62)
    ax.legend(loc="lower left", fontsize=13)
    ax.set_title("(a) the guarantee, as a function of the step")
    S.square(ax)

    # ---- (b) three step sizes on a 1-D quadratic ---------------------
    ax = axes[1]
    lam = 1.0
    w = np.linspace(-2.75, 2.75, 400)
    ax.plot(w, 0.5 * lam * w ** 2, color=S.GREY, lw=1.8, zorder=1)

    settings = [(0.10, S.L2, "too small"),
                (0.85, S.L1, "about right"),
                (2.10, S.OUTC, "too big")]
    for eta_, col, lab in settings:
        x, ys, xs = -2.0, [], []
        for _ in range(9 if eta_ < 2.0 else 5):
            xs.append(x); ys.append(0.5 * lam * x ** 2)
            x = x - eta_ * lam * x
        xs, ys = np.array(xs), np.array(ys)
        ax.plot(xs, ys, "-o", color=col, lw=1.5, ms=5.2, alpha=0.95,
                label=r"$\eta = %.2f$  (%s)" % (eta_, lab), zorder=3)
    ax.plot(0, 0, "*", ms=15, mfc="white", mec=S.OUTC, mew=1.8, zorder=6)
    ax.set_xlabel("$w$")
    ax.set_ylabel("$E(w)$")
    ax.set_xlim(-2.75, 2.75); ax.set_ylim(-0.22, 4.15)
    ax.legend(loc="upper center", fontsize=12.5, ncol=1)
    ax.set_title("(b) the same three regimes, walked out")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

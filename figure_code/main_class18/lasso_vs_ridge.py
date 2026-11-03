"""lasso_vs_ridge — why q = 1 sets coefficients to zero and q = 2 does not

Used in: Class 18 (B), slide(s): "The two geometries side by side"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python lasso_vs_ridge.py            -> output/lasso_vs_ridge.pdf
         python lasso_vs_ridge.py --out DIR  -> DIR/lasso_vs_ridge.pdf
Ported from: Class18_Regularization_Explicit_Implicit/B_Recreated_Figures/src/figures/fig03_lasso_vs_ridge.py

Following the geometry of Bishop & Bishop (2024), Fig. 9.6.

`lasso_vs_ridge`  the constrained form of the same problem. Minimising
    E(w) subject to sum |w_j|^q <= eta puts the solution where a contour
    of E first touches the constraint region. Against the diamond that
    contact happens at a corner, where w1 = 0 exactly; against the disc
    it happens on a smooth arc, where w1 is merely small.

(Split from fig03_lasso_vs_ridge.py, which also wrote qnorm_contours.pdf.)
"""
import numpy as np
import matplotlib.pyplot as plt
import common as S
from models import rotated_hessian

NAME = "lasso_vs_ridge"

WSTAR = np.array([1.40, 2.40])
HESS, _ = rotated_hessian((1.0, 3.0), np.deg2rad(45.0))   # correlated inputs
ETA = 1.0
LIMS = (-1.5, 3.4, -1.5, 4.2)


def _E(W1, W2):
    D1 = W1 - WSTAR[0]; D2 = W2 - WSTAR[1]
    return 0.5 * (HESS[0, 0] * D1 ** 2 + 2 * HESS[0, 1] * D1 * D2
                  + HESS[1, 1] * D2 ** 2)


def _constrained(q, n=200001):
    """The point of {sum|w|^q <= eta} at which E is smallest."""
    t = np.linspace(0, 2 * np.pi, n)
    c, s = np.cos(t), np.sin(t)
    r = (np.abs(c) ** q + np.abs(s) ** q) ** (-1.0 / q)
    w1, w2 = ETA ** (1.0 / q) * r * c, ETA ** (1.0 / q) * r * s
    k = int(np.argmin(_E(w1, w2)))
    return np.array([w1[k], w2[k]])


def _region(ax, q, col):
    t = np.linspace(0, 2 * np.pi, 4000)
    c, s = np.cos(t), np.sin(t)
    r = (np.abs(c) ** q + np.abs(s) ** q) ** (-1.0 / q)
    ax.fill(ETA ** (1.0 / q) * r * c, ETA ** (1.0 / q) * r * s,
            color=col, alpha=0.16, lw=0, zorder=2)
    ax.plot(ETA ** (1.0 / q) * r * c, ETA ** (1.0 / q) * r * s,
            color=col, lw=2.2, zorder=4)


def build():
    S.use()

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.9))
    x0, x1, y0, y1 = LIMS
    G1, G2 = np.meshgrid(np.linspace(x0, x1, 520), np.linspace(y0, y1, 600))
    E = _E(G1, G2)

    for ax, q, title, eq in (
            (axes[0], 1.0, r"(a) $q = 1$: the lasso",
             r"$|w_1| + |w_2| \leq \eta$"),
            (axes[1], 2.0, r"(b) $q = 2$: weight decay",
             r"$w_1^2 + w_2^2 \leq \eta$")):
        w = _constrained(q)
        Ew = _E(*w)
        # nested error contours fading outwards, the contact one heavier
        inner = (0.06, 0.20, 0.42, 0.68)
        outer = (1.3, 1.65, 2.05, 2.5)
        for f in inner:
            ax.contour(G1, G2, E, levels=[Ew * f], colors=[S.OUTC],
                       linewidths=1.3, alpha=0.95)
        ax.contour(G1, G2, E, levels=[Ew], colors=[S.OUTC], linewidths=2.4)
        for k, f in enumerate(outer):
            ax.contour(G1, G2, E, levels=[Ew * f], colors=[S.OUTC],
                       linewidths=1.3, alpha=0.75 - 0.15 * k)
        # the constraint region, filled, with its equation inside
        t = np.linspace(0, 2 * np.pi, 4000)
        c, s_ = np.cos(t), np.sin(t)
        r = (np.abs(c) ** q + np.abs(s_) ** q) ** (-1.0 / q)
        bx, by = ETA ** (1.0 / q) * r * c, ETA ** (1.0 / q) * r * s_
        ax.fill(bx, by, color="#5DBB78", alpha=1.0, lw=0, zorder=3)
        ax.plot(bx, by, color="#1E7A3A", lw=1.6, zorder=4)
        ax.annotate(eq, xy=(0.0, -1.30), ha="center", fontsize=16,
                    color="black", zorder=6)
        # the two minima
        ax.plot(*WSTAR, "o", ms=7, mfc=S.OUTC, mec="white", mew=1.2, zorder=7)
        ax.annotate(r"$\mathbf{w}^\star$", xy=WSTAR + np.array([0.10, 0.06]),
                    color=S.OUTC, fontsize=16)
        ax.plot(*w, "o", ms=8.5, mfc="white", mec=S.L1, mew=2.2, zorder=8)
        ax.annotate(r"$\hat{\mathbf{w}} = (%.2f,\ %.2f)$" % (w[0], w[1]),
                    xy=(w[0] + 0.55, w[1] - 0.42), color=S.L1, fontsize=14.5,
                    ha="left", zorder=9)
        ax.annotate(r"$E(\mathbf{w})$", xy=(2.75, 3.75), color=S.OUTC,
                    fontsize=17)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_aspect("equal"); ax.axis("off")
        ax.annotate("$w_1$", xy=(x1 - 0.42, -0.34), fontsize=16)
        ax.annotate("$w_2$", xy=(0.10, y1 - 0.32), fontsize=16)
        ax.annotate("", xy=(x1, 0), xytext=(x0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
        ax.annotate("", xy=(0, y1), xytext=(0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
        ax.set_title(title, y=-0.09)
        print("q = %g: contact at (%.2f, %.2f)" % (q, w[0], w[1]))

    fig.tight_layout(w_pad=2.2)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

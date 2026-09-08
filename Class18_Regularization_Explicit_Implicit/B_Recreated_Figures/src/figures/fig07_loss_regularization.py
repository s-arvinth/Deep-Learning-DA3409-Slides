"""What a penalty does to a loss surface with several minima.

The layout of Prince (2023), Fig. 9.1, for a model of this deck's own:
yhat(x) = sin(w0 x) + 0.5 sin(w1 x), fitted by least squares to 40
points generated from itself at w* = (3, 7).  Fitting a frequency is
non-convex -- aliases and the two terms swapping roles give the loss
many local minima -- so the surface has the structure Prince's does.

(a) The loss.  Every local minimum found by a grid search followed by
    gradient descent is marked; the global one is the star.
(b) The penalty (lambda/2)||w||^2, a bowl centred on the origin.
(c) Their sum.  Fewer minima survive, and the global one has moved
    towards the origin.

Nothing is drawn by hand: the minima are located numerically, and the
counts printed in the titles are what the search found.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common import models as M

NAME = "loss_regularization"

LIM = 8.0
LAM = 0.004
NG = 321


def _minima(fun, F, g):
    pts = M.local_minima(F, g, g, lambda c: M.descend(fun, c, steps=300))
    pts = [p for p in pts if np.all(np.abs(p) < LIM)]
    vals = np.array([fun(p) for p in pts])
    order = np.argsort(vals)
    return [pts[k] for k in order], vals[order]


def build():
    S.use()
    x, y = M.mm_data()
    g = np.linspace(-LIM, LIM, NG)
    W0, W1 = np.meshgrid(g, g)
    Lg = M.mm_loss_grid(W0, W1, x, y)
    Rg = 0.5 * LAM * (W0 ** 2 + W1 ** 2)
    Tg = Lg + Rg

    f_loss = lambda w: M.mm_loss(w, x, y)
    f_tot = lambda w: M.mm_loss(w, x, y) + 0.5 * LAM * float(w @ w)
    pl, vl = _minima(f_loss, Lg, g)
    pt, vt = _minima(f_tot, Tg, g)
    print("minima of L: %d, of L + reg: %d" % (len(pl), len(pt)))
    print("global minimum moved from (%.2f, %.2f) to (%.2f, %.2f)"
          % (pl[0][0], pl[0][1], pt[0][0], pt[0][1]))
    assert len(pt) < len(pl)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))
    panels = ((Lg, "(a) loss $E(\\mathbf{w})$", pl),
              (Rg, r"(b) penalty $\frac{\lambda}{2}\|\mathbf{w}\|^2$", None),
              (Tg, "(c) loss $+$ penalty", pt))
    for ax, (Z, title, pts) in zip(axes, panels):
        ax.contourf(W0, W1, np.log10(Z + 1e-3), levels=26, cmap=S.SEQ,
                    alpha=0.92)
        ax.contour(W0, W1, np.log10(Z + 1e-3), levels=10, colors="white",
                   linewidths=0.5, alpha=0.45)
        if pts is not None:
            P = np.array(pts)
            ax.plot(P[1:, 0], P[1:, 1], "o", ms=8, mfc=S.L2, mec="white",
                    mew=1.4, zorder=6, label="local minimum")
            ax.plot(P[0, 0], P[0, 1], "*", ms=17, mfc="white", mec=S.OUTC,
                    mew=1.9, zorder=7, label="global minimum")
            ax.set_title(title + "   (%d minima)" % len(pts), fontsize=15)
        else:
            ax.set_title(title, fontsize=15)
        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
        ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
        ax.set_xticks([-8, -4, 0, 4, 8]); ax.set_yticks([-8, -4, 0, 4, 8])
        S.square(ax)

    # the move of the global minimum, drawn on (c)
    ax = axes[2]
    ax.annotate("", xy=tuple(pt[0]), xytext=tuple(pl[0]),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.6),
                zorder=8)
    ax.plot(*pl[0], "*", ms=13, mfc="none", mec="white", mew=1.2, zorder=7)
    ax.annotate("was here", xy=tuple(pl[0]), xytext=(pl[0][0] - 4.9,
                pl[0][1] - 0.3), color="white", fontsize=12,
                arrowprops=dict(arrowstyle="-", color="white", lw=0.8))
    axes[0].legend(loc="lower left", fontsize=11.5, framealpha=0.9)
    fig.tight_layout(w_pad=1.4)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

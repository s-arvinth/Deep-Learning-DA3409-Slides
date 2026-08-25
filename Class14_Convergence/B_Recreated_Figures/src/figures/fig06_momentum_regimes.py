"""The two regimes, drawn on the error curve itself.

Following the geometry of Bishop & Bishop (2024), Figs. 7.4 and 7.5.

Panel (a): a direction of LOW curvature. The gradient is small, every
step points the same way, and the iterate creeps towards the minimum.
Successive displacements Delta w are aligned, so a momentum term adds
them and the effective learning rate grows to eta / (1 - mu).

Panel (b): a direction of HIGH curvature, with the same learning rate.
The step overshoots, so successive displacements ALTERNATE in sign. A
momentum term now cancels them and the effective rate stays below eta.

Same eta in both panels: the asymmetry is a property of the curvature,
not of the tuning.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S

NAME = "momentum_regimes"

ETA = 0.20
LAM_LO, LAM_HI = 1.50, 9.10          # eta*lam = 0.30 and 1.82
MU = 0.9


def _iterates(lam, w0, n):
    w, out = w0, [w0]
    for _ in range(n):
        w = w - ETA * lam * w
        out.append(w)
    return np.array(out)


def _panel(ax, lam, w0, n, span, rows, title, note, note_col):
    w = _iterates(lam, w0, n)
    t = np.linspace(-span, span, 400)
    E = 0.5 * lam * t ** 2
    top = 0.5 * lam * span ** 2

    ax.plot(t, E, color=S.L1, lw=2.6, zorder=3)
    ax.axvline(0.0, color=S.GREY, lw=1.0, ls=(0, (3, 3)), zorder=1)

    # the iterates, on the curve
    ax.plot(w, 0.5 * lam * w ** 2, "o", ms=8, mfc="white", mec=S.OUTC,
            mew=2.0, zorder=6)
    for k, wk in enumerate(w):
        ax.annotate(r"$\mathbf{w}^{(%d)}$" % k,
                    xy=(wk, 0.5 * lam * wk ** 2 + 0.055 * top),
                    color=S.OUTC, fontsize=12.5, ha="center", va="bottom")

    # the displacements, on staggered rows beneath the curve
    for k in range(n):
        y = -rows[k % len(rows)] * top
        ax.plot([w[k], w[k]], [0.5 * lam * w[k] ** 2, y], color=S.GREY,
                lw=0.8, ls=(0, (2, 2.5)), zorder=2)
        ax.annotate("", xy=(w[k + 1], y), xytext=(w[k], y),
                    arrowprops=dict(arrowstyle="-|>", color=S.L2, lw=2.2),
                    zorder=5)
        ax.annotate(r"$\Delta \mathbf{w}^{(%d)}$" % k,
                    xy=(0.5 * (w[k] + w[k + 1]), y - 0.035 * top),
                    color=S.L2, fontsize=12.5, ha="center", va="top")

    ax.annotate(note, xy=(0.5, 0.955), xycoords="axes fraction",
                color=note_col, fontsize=14.5, ha="center", va="top")
    ax.set_xlim(-span * 1.12, span * 1.12)
    ax.set_ylim(-max(rows) * top - 0.16 * top, 1.20 * top)
    ax.set_xlabel(r"$w$  along this direction")
    ax.set_ylabel(r"$E(\mathbf{w})$")
    ax.set_title(title)
    ax.set_box_aspect(0.80)


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))

    _panel(axes[0], LAM_LO, -2.72, 3, 3.0, (0.20, 0.36),
           r"(a) low curvature:  $\eta\lambda = %.2f$" % (ETA * LAM_LO),
           "every step points the same way\n"
           r"momentum adds them:  $\eta \to \eta/(1-\mu) = %.0f\,\eta$"
           % (1 / (1 - MU)),
           S.OUTC)

    _panel(axes[1], LAM_HI, -1.30, 3, 1.75, (0.22, 0.40, 0.58),
           r"(b) high curvature:  $\eta\lambda = %.2f$" % (ETA * LAM_HI),
           "successive steps alternate\n"
           r"momentum cancels them:  $\eta \to \eta/(1+\mu) = %.2f\,\eta$"
           % (1 / (1 + MU)),
           S.OUTC)

    fig.tight_layout(w_pad=1.6)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

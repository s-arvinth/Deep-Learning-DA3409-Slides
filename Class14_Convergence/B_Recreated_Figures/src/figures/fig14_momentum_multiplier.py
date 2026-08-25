"""How much momentum adds in one regime and cancels in the other.

The companion to fig06, which shows the two regimes geometrically. Here
the same two regimes are measured: the length of the accumulated step,
in units of eta * ||grad E||.

Panel (a): with the gradient pointing the same way each time the
multiplier is a partial geometric series and climbs to 1 / (1 - mu).

Panel (b): with the gradient alternating, the terms cancel and the
multiplier settles at 1 / (1 + mu) -- below one, not above. Note the
vertical scale.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S

NAME = "momentum_multiplier"

MU = 0.9


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))

    # ---- (a) aligned gradients: the sum of a geometric series --------
    ax = axes[0]
    T = np.arange(1, 41)
    eff = (1.0 - MU ** T) / (1.0 - MU)
    ax.plot(T, eff, color=S.OUTC, lw=2.6)
    ax.axhline(1.0 / (1.0 - MU), color=S.L1, lw=1.8, ls=(0, (5, 3)))
    ax.annotate(r"$\dfrac{1}{1-\mu} = %.0f$" % (1 / (1 - MU)),
                xy=(24, 1 / (1 - MU) - 1.7), fontsize=16, color=S.L1)
    ax.axhline(1.0, color=S.GREY, lw=1.2, ls=(0, (3, 3)))
    ax.annotate("no momentum", xy=(24, 1.45), fontsize=13.5, color=S.GREY)
    ax.set_xlabel("consecutive aligned steps")
    ax.set_ylabel(r"step length $\,/\,\eta\|\nabla E\|$")
    ax.set_ylim(0, 12)
    ax.set_title(r"(a) aligned: $\eta \to \eta/(1-\mu)$")
    S.square(ax)

    # ---- (b) alternating gradients: the terms cancel ------------------
    ax = axes[1]
    sgn = (-1.0) ** np.arange(60)
    dw, out = 0.0, []
    for k in range(60):
        dw = -sgn[k] + MU * dw
        out.append(abs(dw))
    ax.plot(np.arange(1, 61), out, color=S.OUTC, lw=2.6)
    ax.axhline(1.0, color=S.GREY, lw=1.2, ls=(0, (3, 3)))
    ax.annotate("no momentum", xy=(30, 1.06), fontsize=13.5, color=S.GREY)
    ax.axhline(1.0 / (1.0 + MU), color=S.L1, lw=1.8, ls=(0, (5, 3)))
    ax.annotate(r"$\dfrac{1}{1+\mu} = %.2f$" % (1 / (1 + MU)),
                xy=(30, 1 / (1 + MU) - 0.30), fontsize=15, color=S.L1)
    ax.set_xlabel("consecutive alternating steps")
    ax.set_ylabel(r"step length $\,/\,\eta\|\nabla E\|$")
    ax.set_ylim(0, 1.6)
    ax.set_title(r"(b) alternating: the terms cancel"
                 "\n" r"(note the vertical scale)", fontsize=15.5)
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

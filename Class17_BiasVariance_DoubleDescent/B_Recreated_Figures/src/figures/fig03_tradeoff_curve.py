"""The classical trade-off, measured two ways.

Both books plot the same three curves against a knob that controls
complexity; they differ only in which knob.

(a) Against capacity K, as in Prince (2023), Fig. 8.9: bias falls,
    variance rises, and the sum has an interior minimum.

(b) Against the penalty ln(lambda) at fixed capacity, as in Bishop &
    Bishop (2024), Fig. 4.8: the same picture, read right to left. The
    excess risk, measured directly and without using the decomposition,
    is overlaid; it lies on top of bias^2 + variance, which is the
    theorem being checked rather than assumed.

Every value is a Monte-Carlo average over L data sets, computed with
Eqs. 4.51 and 4.52 of Bishop & Bishop (2024).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import truth, sample, piecewise_fit, bias_variance

NAME = "tradeoff_curve"

GRID = np.linspace(0.02, 0.98, 220)
N, L = 60, 200


def _excess_risk(knob, sweep, K=20, seed=5, reps=400):
    """The excess risk E_D[ mean_x (f - h)^2 ], measured directly.

    This is the quantity the decomposition claims to equal, so plotting
    it beside bias^2 + variance is a check on the theorem rather than a
    restatement of it.
    """
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(reps):
        x, y = sample(N, rng)
        if sweep == "capacity":
            f = piecewise_fit(x, y, int(knob), grid=GRID)
        else:
            f = piecewise_fit(x, y, K, lam=knob, grid=GRID)
        tot += np.mean((f - truth(GRID)) ** 2)
    return tot / reps


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))

    # ---- (a) sweep the capacity --------------------------------------
    ax = axes[0]
    Ks = np.arange(2, 25)
    b2 = np.empty(len(Ks)); vr = np.empty(len(Ks))
    for i, K in enumerate(Ks):
        b2[i], vr[i] = bias_variance(K, GRID, n=N, L=L, seed=7,
                                     sweep="capacity")
    tot = b2 + vr
    ax.plot(Ks, b2, color=S.OUTC, lw=2.5, label=r"bias$^2$")
    ax.plot(Ks, vr, color=S.L2, lw=2.5, label="variance")
    ax.plot(Ks, tot, color=S.L1, lw=2.5, ls=(0, (5, 2.5)),
            label=r"bias$^2$ + variance")
    kstar = Ks[int(np.argmin(tot))]
    ax.axvline(kstar, color=S.GREY, lw=1.1, ls=(0, (3, 3)))
    ax.annotate(r"$K^\star = %d$" % kstar, xy=(kstar + 0.8, 0.020),
                color=S.GREY, fontsize=14)
    ax.set_xlabel("capacity $K$  (basis functions)")
    ax.set_ylabel("error")
    ax.set_ylim(0, 0.20)
    ax.legend(loc="upper right", fontsize=13, bbox_to_anchor=(0.99, 0.72))
    ax.set_title("(a) against capacity")
    S.square(ax)

    # ---- (b) sweep the penalty ---------------------------------------
    ax = axes[1]
    lnl = np.linspace(-12.0, 3.5, 24)
    b2 = np.empty(len(lnl)); vr = np.empty(len(lnl))
    for i, u in enumerate(lnl):
        b2[i], vr[i] = bias_variance(np.exp(u), GRID, n=N, L=L, seed=7,
                                     sweep="penalty", K=20)
    tot = b2 + vr
    te = np.array([_excess_risk(np.exp(u), "penalty") for u in lnl])
    ax.plot(lnl, b2, color=S.OUTC, lw=2.5, label=r"bias$^2$")
    ax.plot(lnl, vr, color=S.L2, lw=2.5, label="variance")
    ax.plot(lnl, tot, color=S.L1, lw=2.5, ls=(0, (5, 2.5)),
            label=r"bias$^2$ + variance")
    ax.plot(lnl, te, color=S.GREY, lw=2.2, ls=(0, (1.5, 2)),
            label="measured excess risk")
    ustar = lnl[int(np.argmin(tot))]
    ax.axvline(ustar, color=S.GREY, lw=1.1, ls=(0, (3, 3)))
    ax.set_xlabel(r"$\ln \lambda$")
    ax.set_ylabel("error")
    ax.set_ylim(0, 0.16)
    ax.legend(loc="upper right", fontsize=12, bbox_to_anchor=(0.99, 0.99))
    ax.set_title("(b) against the penalty")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

"""Dropout, measured on the same 1-D problem.

Dropout zeroes a random subset of units at every step, so each update is
computed on a different thinned network. Two consequences are visible
here.

(a) The fitted function. Without dropout the model is free to build a
    feature that only matters between two training points, because
    nothing in the loss objects. With dropout that feature is absent
    from half the updates, so whatever it was compensating for reappears
    and the next step removes it. What survives is smoother, which is
    the mechanism Prince (2023), Fig. 9.9 describes.

(b) Test error against the dropout probability, six runs at each rate,
    with the spread shaded. On this problem the best rate is 0.1, not
    the 0.5 that is usual for hidden units in a large network -- there
    are only 44 features here and very little redundancy to spare. The
    usual defaults are calibrated on much wider networks, and this is a
    reminder that they are defaults and not results. Beyond p = 0.8 the
    1/(1-p) rescaling makes the updates diverge, so the sweep stops.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import truth, sample, features, sgd_path

NAME = "dropout"

GRID = np.linspace(0.02, 0.98, 500)
N, K = 26, 44
STEPS, LR = 26000, 0.030
RATES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
REPS = 6


def _run(x, y, cen, drop, rng):
    Phi, _ = features(x, K, centres=cen)
    w, _ = sgd_path(Phi, y, STEPS, LR, rng, drop=drop)
    return w


def build():
    S.use()
    rng = np.random.default_rng(3)
    x, y = sample(N, rng)
    xt, yt = sample(600, rng)
    _, cen = features(x, K)
    Pg, _ = features(GRID, K, centres=cen)
    Pt, _ = features(xt, K, centres=cen)
    h = truth(GRID)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))

    # ---- (a) two fits -------------------------------------------------
    ax = axes[0]
    ax.plot(GRID, h, color=S.GREY, lw=2.4, zorder=3, label=r"$h(x)$")
    for drop, col, lab in ((0.0, S.L1, "no dropout"),
                           (0.5, S.OUTC, r"dropout, $p = 0.5$")):
        w = _run(x, y, cen, drop, np.random.default_rng(21))
        ax.plot(GRID, Pg @ w, color=col, lw=2.5, zorder=4, label=lab)
    ax.plot(x, y, "o", ms=6, mfc="white", mec=S.L1, mew=1.6, zorder=6)
    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(-2.3, 2.3)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc="lower left", fontsize=12.5)
    ax.set_title("(a) the same model, thinned while training")
    S.square(ax)

    # ---- (b) the rate -------------------------------------------------
    ax = axes[1]
    err = np.empty(len(RATES)); sd = np.empty(len(RATES))
    for i, p in enumerate(RATES):
        e = [np.mean((Pt @ _run(x, y, cen, p, np.random.default_rng(100 + r))
                      - yt) ** 2) for r in range(REPS)]
        err[i] = np.mean(e); sd[i] = np.std(e)
    ax.fill_between(RATES, np.maximum(err - sd, 1e-4), err + sd,
                    color=S.OUTC, alpha=0.16, lw=0)
    ax.semilogy(RATES, err, "o-", color=S.OUTC, lw=2.5, ms=6)
    k = int(np.argmin(err))
    ax.plot(RATES[k], err[k], "*", ms=17, mfc="white", mec=S.L1, mew=1.9,
            zorder=6)
    ax.annotate(r"best at $p = %.1f$" % RATES[k],
                xy=(RATES[k], err[k] * 0.72), color=S.L1, fontsize=13,
                ha="center", va="top")
    ax.set_xlabel("dropout probability $p$")
    ax.set_ylabel("test error")
    ax.set_title("(b) the best rate is problem-dependent")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

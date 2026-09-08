"""Training error falls; validation error does not.

Following the diagnostic of Bishop & Bishop (2024), Fig. 9.7 and the
sequence of Prince (2023), Fig. 9.6.

(a) The two curves for one run of SGD on an over-capacity model. The
    training error decreases throughout. The validation error falls,
    reaches a minimum, and then climbs as the model starts to fit the
    noise. The dashed line marks the iteration to stop at, and it is
    found by watching, not by predicting.

(b) The fitted function at three points along the same run: before the
    minimum, at it, and long after. Early it has the coarse shape and
    not the detail; at the minimum it is close to h(x); later it is
    chasing individual observations.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import truth, sample, features, sgd_path, SIGMA

NAME = "learning_curves"

GRID = np.linspace(0.02, 0.98, 500)
N, K = 24, 40
STEPS, LR = 40000, 0.055
SNAPS = (15, 60, 40000)


def build():
    S.use()
    rng = np.random.default_rng(4)
    x, y = sample(N, rng)
    xv, yv = sample(400, rng)

    Phi, cen = features(x, K)
    Pv, _ = features(xv, K, centres=cen)
    Pg, _ = features(GRID, K, centres=cen)

    rec = sorted(set(list(np.unique(np.round(np.logspace(0.6, np.log10(STEPS),
                                                         70)).astype(int)))
                     + list(SNAPS)))
    _, snaps = sgd_path(Phi, y, STEPS, LR, rng, record=set(rec))
    ts = np.array(sorted(snaps))
    tr = np.array([np.mean((Phi @ snaps[t] - y) ** 2) for t in ts])
    va = np.array([np.mean((Pv @ snaps[t] - yv) ** 2) for t in ts])
    tstar = ts[int(np.argmin(va))]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))

    ax = axes[0]
    ax.loglog(ts, tr, color=S.L2, lw=2.5, label="training error")
    ax.loglog(ts, va, color=S.OUTC, lw=2.5, label="validation error")
    ax.axhline(SIGMA ** 2, color=S.GREY, lw=1.1, ls=(0, (2, 3)))
    ax.annotate(r"noise floor $\sigma^2$", xy=(3.0e3, SIGMA ** 2 * 1.30),
                color=S.GREY, fontsize=12.5)
    ax.axvline(tstar, color=S.GREY, lw=1.3, ls=(0, (4, 3)))
    ax.annotate("stop here\n$\\tau^\\star = %d$" % tstar,
                xy=(tstar * 1.4, 0.86), xycoords=("data", "axes fraction"),
                color=S.GREY, fontsize=13, ha="left", va="top")
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean squared error")
    ax.legend(loc="lower left", fontsize=13)
    ax.set_title("(a) the two curves diverge")
    S.square(ax)

    ax = axes[1]
    ax.plot(GRID, truth(GRID), color=S.GREY, lw=2.4, zorder=3,
            label=r"$h(x)$")
    for t, c, lab in zip(SNAPS, (S.L2, S.OUTC, S.L1),
                         ("too early", r"at $\tau^\star$", "too late")):
        ax.plot(GRID, Pg @ snaps[t], color=c, lw=2.3, zorder=4,
                label=r"%s, $\tau = %d$" % (lab, t))
    ax.plot(x, y, "o", ms=6, mfc="white", mec=S.L1, mew=1.6, zorder=6)
    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(-2.3, 2.3)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc="lower left", fontsize=11.5, handlelength=1.3,
              labelspacing=0.25)
    ax.set_title("(b) the same run, as a function")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

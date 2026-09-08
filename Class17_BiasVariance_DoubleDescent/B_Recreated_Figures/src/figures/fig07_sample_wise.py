"""More data can make the model worse.

Everything so far varied capacity at a fixed sample size. Here capacity
is held fixed at p = 300 and the sample size n is varied instead. Since
gamma = p/n, growing n moves the model from right to left through the
interpolation threshold -- and straight through the peak.

(a) Excess risk against n, measured, with the closed form of
    Hastie et al. (2022) overlaid. Between n = 100 and n = p = 300 the
    curve goes UP: over that range, collecting more data hurts.

(b) The same curve at two capacities. Each has its own peak, at its own
    n = p, so which model is preferable depends entirely on where the
    available n sits relative to both thresholds.

This is the phenomenon Bishop & Bishop (2024), Fig. 9.11 reports for a
transformer, in the one setting where it can be derived.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import minnorm_risk_theory

NAME = "sample_wise"

P = 300
SIGMA = 1.0
REPS = 20
NS = np.array([40, 60, 90, 130, 170, 210, 250, 280, 320, 360, 420,
               500, 620, 800, 1100, 1600, 2400])


def _measure(p, ns, seed=4):
    rng = np.random.default_rng(seed)
    w = np.zeros(p); w[0] = 1.0
    out = np.empty(len(ns))
    for i, n in enumerate(ns):
        acc = np.empty(REPS)
        for r in range(REPS):
            X = rng.normal(size=(n, p))
            Y = X @ w + SIGMA * rng.normal(size=n)
            acc[r] = np.sum((np.linalg.pinv(X) @ Y - w) ** 2)
        out[i] = acc.mean()
    return out


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))

    # ---- (a) one capacity, measured against the closed form ----------
    ax = axes[0]
    risk = _measure(P, NS)
    nl = np.concatenate([np.linspace(30, P / 1.015, 240),
                         np.linspace(P / 0.985, 2400, 240)])
    ax.plot(nl, minnorm_risk_theory(P / nl, SIGMA), color=S.L1, lw=2.3,
            label="Hastie et al. (2022)")
    ax.plot(NS, risk, "o", ms=6.5, mfc="none", mec=S.OUTC, mew=1.8,
            label=r"measured,  $p = %d$" % P)
    ax.axvline(P, color=S.GREY, lw=1.3, ls=(0, (4, 3)))
    ax.annotate("", xy=(280, 9.0), xytext=(110, 1.35),
                arrowprops=dict(arrowstyle="-|>", color=S.OUTC, lw=2.0))
    ax.annotate("more data,\nworse model", xy=(52, 2.6), color=S.OUTC,
                fontsize=13.5, ha="left", va="bottom")
    ax.annotate(r"$n = p$", xy=(P * 1.12, 0.92), color=S.GREY, fontsize=13.5,
                xycoords=("data", "axes fraction"), va="top")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training set size $n$")
    ax.set_ylabel("excess risk")
    ax.set_ylim(2e-2, 2e2)
    ax.legend(loc="lower left", fontsize=12.5)
    ax.set_title("(a) one capacity")
    S.square(ax)

    # ---- (b) two capacities ------------------------------------------
    ax = axes[1]
    for p, col, ls in ((120, S.L2, "-"), (600, S.OUTC, "-")):
        nl = np.concatenate([np.linspace(30, p / 1.02, 240),
                             np.linspace(p / 0.98, 2400, 240)])
        ax.plot(nl, minnorm_risk_theory(p / nl, SIGMA), ls, color=col,
                lw=2.3, label=r"$p = %d$" % p)
        ax.axvline(p, color=col, lw=1.0, ls=(0, (3, 3)), alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training set size $n$")
    ax.set_ylabel("excess risk")
    ax.set_xlim(30, 2400)
    ax.set_ylim(2e-2, 2e2)
    ax.legend(loc="lower left", fontsize=13)
    ax.set_title("(b) the peak moves with capacity")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

"""Why the peak is at the threshold: the one case that can be solved.

For minimum-norm least squares on an isotropic Gaussian design the
excess risk has a closed form (Hastie et al., 2022, Thm. 1), so the peak
is not an empirical curiosity but a divergence that can be pointed at.

(a) Measured risk against that closed form, over four decades. They
    agree everywhere, including on both sides of gamma = 1.

(b) The mechanism. The squared norm of the fitted parameter vector
    peaks at exactly the same place. At gamma = 1 there is precisely one
    solution that interpolates, and it is forced to be enormous; on
    either side there is either slack (gamma < 1) or a whole subspace of
    interpolants to pick the smallest from (gamma > 1).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import minnorm_risk_theory

NAME = "minnorm_theory"

N = 200
SIGMA = 1.0
REPS = 24
GAMMAS = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.96,
                   1.04, 1.1, 1.2, 1.4, 1.7, 2.2, 3.0, 4.5, 7.0])


def _run(seed=0):
    rng = np.random.default_rng(seed)
    risk = np.empty(len(GAMMAS)); nrm = np.empty(len(GAMMAS))
    for i, g in enumerate(GAMMAS):
        p = max(int(round(g * N)), 1)
        w = np.zeros(p); w[0] = 1.0
        rk = np.empty(REPS); nm = np.empty(REPS)
        for r in range(REPS):
            X = rng.normal(size=(N, p))
            Y = X @ w + SIGMA * rng.normal(size=N)
            wh = np.linalg.pinv(X) @ Y
            rk[r] = np.sum((wh - w) ** 2)
            nm[r] = np.sum(wh ** 2)
        risk[i] = rk.mean(); nrm[i] = nm.mean()
    return risk, nrm


def build():
    S.use()
    risk, nrm = _run()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))

    # ---- (a) measured against the closed form ------------------------
    ax = axes[0]
    for lo, hi in ((0.02, 0.985), (1.015, 7.5)):
        g = np.linspace(lo, hi, 300)
        ax.plot(g, minnorm_risk_theory(g, SIGMA), color=S.L1, lw=2.4,
                label="Hastie et al. (2022)" if lo < 1 else None)
    ax.plot(GAMMAS, risk, "o", ms=7, mfc="none", mec=S.OUTC, mew=1.9,
            label=r"measured,  $n = %d$" % N)
    ax.axvline(1.0, color=S.GREY, lw=1.3, ls=(0, (4, 3)))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\gamma = p/n$")
    ax.set_ylabel("excess risk")
    ax.set_ylim(2e-2, 4e2)
    ax.legend(loc="lower left", fontsize=13)
    ax.set_title("(a) the risk, in closed form")
    S.square(ax)

    # ---- (b) the norm of the solution --------------------------------
    ax = axes[1]
    ax.plot(GAMMAS, nrm, "-o", color=S.L2, lw=2.4, ms=5.2)
    ax.axvline(1.0, color=S.GREY, lw=1.3, ls=(0, (4, 3)))
    ax.axhline(1.0, color=S.GREY, lw=1.0, ls=(0, (2, 3)))
    ax.annotate(r"$\|\mathbf{w}\|^2 = 1$", xy=(0.055, 1.30), color=S.GREY,
                fontsize=13.5, ha="left")
    ax.annotate("one interpolant,\nand it is huge", xy=(1.0, nrm.max()),
                xytext=(0.055, 0.30 * nrm.max()), color=S.OUTC, fontsize=13,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=S.OUTC, lw=1.4))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\gamma = p/n$")
    ax.set_ylabel(r"$\|\hat{\mathbf{w}}\|^2$")
    ax.set_title("(b) the mechanism: the norm blows up")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

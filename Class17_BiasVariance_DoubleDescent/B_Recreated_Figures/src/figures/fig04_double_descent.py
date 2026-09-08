"""Double descent, measured on a random-feature model.

The set-up is the smallest one that reproduces what Bishop & Bishop
(2024), Fig. 9.9 and Prince (2023), Fig. 8.10 report for deep networks:

    latent  z in R^15,        n = 150 training points
    target  Y = z . w  + eps,  eps ~ N(0, 0.55^2)
    model   p features  tanh(A z + b),  A and b drawn once and frozen,
            fitted by minimum-norm least squares

Capacity is p, and only p is varied. Nothing is regularised and nothing
is stopped early, so every effect below is a property of capacity alone.

(a) Training and test error against p. The training error reaches
    exactly zero at p = n and stays there. The test error rises to a
    sharp peak at the same place.

(b) The test error alone, with the three regimes marked. The minimum in
    the over-parameterised regime is LOWER than the classical one, which
    is the observation that the bias-variance trade-off does not predict.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S

NAME = "double_descent"

D, N, NT = 15, 150, 2500
SIGMA = 0.55
REPS = 8
PS = np.array([5, 10, 20, 30, 45, 70, 100, 125, 140, 150, 165, 195,
               240, 320, 450, 700, 1100, 1800, 3000, 6000])


def _run(seed=1):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=D); w /= np.linalg.norm(w)
    Z = rng.normal(size=(N, D)); Zt = rng.normal(size=(NT, D))
    f, ft = Z @ w, Zt @ w
    Y = f + SIGMA * rng.normal(size=N)

    train = np.empty(len(PS)); test = np.empty(len(PS))
    for i, p in enumerate(PS):
        tr = np.empty(REPS); te = np.empty(REPS)
        for r in range(REPS):
            A = rng.normal(size=(D, p)) / np.sqrt(D)
            b = 0.5 * rng.normal(size=p)
            P = np.tanh(Z @ A + b); Pt = np.tanh(Zt @ A + b)
            wh = np.linalg.pinv(P) @ Y
            tr[r] = np.mean((Y - P @ wh) ** 2)
            te[r] = np.mean((ft - Pt @ wh) ** 2)
        train[i] = tr.mean(); test[i] = te.mean()
    return train, test


def build():
    S.use()
    train, test = _run()
    lo = PS < N                       # classical regime
    hi = PS > N                       # modern regime
    c_i = int(np.argmin(np.where(lo, test, np.inf)))
    m_i = int(np.argmin(np.where(hi, test, np.inf)))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))

    # ---- (a) both curves ---------------------------------------------
    ax = axes[0]
    ax.axvline(N, color=S.GREY, lw=1.3, ls=(0, (4, 3)), zorder=2)
    ax.plot(PS, np.maximum(train, 1.5e-4), "-o", color=S.L2, lw=2.3, ms=4.2,
            label="training error")
    ax.plot(PS, test, "-o", color=S.OUTC, lw=2.3, ms=4.2,
            label="test error")
    ax.annotate("interpolation\nthreshold  $p = n$", xy=(N * 1.25, 0.80),
                xycoords=("data", "axes fraction"), color=S.GREY,
                fontsize=13, ha="left", va="top")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("capacity  $p$   (features)")
    ax.set_ylabel("mean squared error")
    ax.set_ylim(1e-4, 1e3)
    ax.legend(loc="lower left", fontsize=13)
    ax.set_title("(a) training and test error")
    S.square(ax)

    # ---- (b) the three regimes ---------------------------------------
    ax = axes[1]
    ax.axvspan(PS[0], 0.72 * N, color=S.L1, alpha=0.07, lw=0)
    ax.axvspan(0.72 * N, 1.45 * N, color=S.OUTC, alpha=0.09, lw=0)
    ax.axvspan(1.45 * N, PS[-1], color=S.L2, alpha=0.07, lw=0)
    ax.plot(PS, test, "-o", color=S.OUTC, lw=2.4, ms=4.6)
    ax.axvline(N, color=S.GREY, lw=1.3, ls=(0, (4, 3)))

    ax.axhline(test[c_i], color=S.GREY, lw=1.1, ls=(0, (2, 3)))
    for i, lab, dy, va in ((c_i, "classical minimum", 0.52, "top"),
                           (m_i, "modern minimum", 2.1, "bottom")):
        ax.plot(PS[i], test[i], "*", ms=17, mfc="white", mec=S.L1, mew=1.8,
                zorder=6)
        ax.annotate("%s\n%.3f" % (lab, test[i]), xy=(PS[i], test[i] * dy),
                    color=S.L1, fontsize=12.5, ha="center", va=va)

    for xpos, lab in ((0.18, "classical"), (0.475, "critical"),
                      (0.80, "modern")):
        ax.annotate(lab, xy=(xpos, 0.965), xycoords="axes fraction",
                    ha="center", va="top", fontsize=12.5, color=S.GREY)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("capacity  $p$   (features)")
    ax.set_ylabel("test error")
    ax.set_ylim(4e-2, 4e3)
    ax.set_title("(b) three regimes, not two")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

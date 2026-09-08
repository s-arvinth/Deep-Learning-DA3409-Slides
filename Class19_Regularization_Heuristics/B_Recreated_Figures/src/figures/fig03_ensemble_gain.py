"""What averaging models actually buys.

Bishop & Bishop (2024), Eq. 9.50 says that if the members' errors have
zero mean and are uncorrelated, the committee error is the average error
divided by M. Eq. 9.49's assumption is the whole content of the claim,
and in practice it fails, so both cases are measured here.

(a) Twelve models fitted to bootstrap resamples of one data set, and
    their mean. The individual curves disagree most where the data are
    thin, and the disagreement is what the average removes.

(b) The committee error against the number of members, on the same
    problem, beside the 1/M law. The measured curve flattens well above
    it: the members are trained on overlapping resamples of one data
    set, so their errors are correlated and most of the promised factor
    never arrives. What does hold is the bound of Exercise 9.15,
    E_COM <= E_AV, drawn as the upper line.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import truth, sample, features, bagged_ensemble

NAME = "ensemble_gain"

GRID = np.linspace(0.03, 0.97, 400)
N, K = 34, 11
LMAX = 64
SHOW = 12


def build():
    S.use()
    rng = np.random.default_rng(11)
    x, y = sample(N, rng)
    h = truth(GRID)
    F = bagged_ensemble(x, y, K, LMAX, rng, GRID)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))

    # ---- (a) the members and their mean -------------------------------
    ax = axes[0]
    for l in range(SHOW):
        ax.plot(GRID, F[l], color=S.L2, lw=0.9, alpha=0.42, zorder=3)
    ax.plot(GRID, F.mean(axis=0), color=S.OUTC, lw=2.9, zorder=5,
            label="committee")
    ax.plot(GRID, h, color=S.GREY, lw=2.4, zorder=4, label=r"$h(x)$")
    ax.plot(x, y, "o", ms=6, mfc="white", mec=S.L1, mew=1.6, zorder=6)
    ax.set_xlim(0.03, 0.97)
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc="lower left", fontsize=13)
    ax.set_title("(a) twelve members and their mean")
    S.square(ax)

    # ---- (b) the gain against the promise -----------------------------
    ax = axes[1]
    Ms = np.arange(1, LMAX + 1)
    e_av = np.mean([np.mean((F[l] - h) ** 2) for l in range(LMAX)])
    e_com = np.array([np.mean((F[:m].mean(axis=0) - h) ** 2) for m in Ms])

    ax.loglog(Ms, e_com, "o-", color=S.OUTC, lw=2.4, ms=4.6,
              label="measured committee error")
    ax.loglog(Ms, e_av / Ms, color=S.L2, lw=2.2, ls=(0, (4, 2.5)),
              label=r"$E_{\mathrm{AV}}/M$, if uncorrelated")
    ax.axhline(e_av, color=S.GREY, lw=1.4, ls=(0, (2, 3)))
    ax.annotate(r"$E_{\mathrm{AV}}$", xy=(1.15, e_av * 1.12), color=S.GREY,
                fontsize=13.5)
    ax.annotate("correlated errors:\nthe gain stops here",
                xy=(LMAX, e_com[-1]), xytext=(6.0, e_com[-1] * 2.6),
                color=S.OUTC, fontsize=12.5, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-|>", color=S.OUTC, lw=1.4))
    ax.set_xlabel("members $M$")
    ax.set_ylabel("squared error against $h$")
    ax.legend(loc="lower left", fontsize=12)
    ax.set_title("(b) the promise, and what arrives")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

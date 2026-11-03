"""early_vs_decay — early stopping and weight decay as the same filter on the eigen-directions

Used in: Class 19 (B), slide(s): "The two filters, measured"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python early_vs_decay.py            -> output/early_vs_decay.pdf
         python early_vs_decay.py --out DIR  -> DIR/early_vs_decay.pdf
Ported from: Class19_Regularization_Heuristics/B_Recreated_Figures/src/figures/fig02_early_vs_decay.py

Early stopping and weight decay are doing the same thing.

On a quadratic error with Hessian eigenvalues lambda_i, both methods
return a scaled version of the unregularised minimum. They differ only
in the scaling:

    gradient descent, tau steps :  ( 1 - (1 - eta lambda_i)^tau ) w*_i
    weight decay, strength alpha:  ( lambda_i / (lambda_i + alpha) ) w*_i

Both are near 1 when lambda_i is large and near 0 when it is small, and
both cross over at the same place once alpha = 1/(eta tau) -- which is
the correspondence Bishop & Bishop (2024), Section 9.3.1 states.

`early_vs_decay`  two panels:

(a) The two filters, for three run lengths and their matched penalties.
    Each pair crosses over together; neither is uniformly larger.

(b) The consequence: the number of directions each method leaves alive,
    sum_i (filter_i), against run length on one axis and against the
    matched penalty on the other. The two curves lie on top of each
    other, so a run length really is a penalty strength.

(Split from fig02_early_vs_decay.py, which also wrote early_stopping_path.pdf.)
"""
import numpy as np
import matplotlib.pyplot as plt
import common as S
from models import gd_filter, ridge_filter, matched_alpha

NAME = "early_vs_decay"

ETA = 0.05
TAUS = (30, 300, 3000)
LAM = np.logspace(-4.0, 1.0, 500)
SPEC = np.logspace(-3.0, 0.6, 120)      # a synthetic eigenvalue spectrum


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))

    # ---- (a) the two filters ------------------------------------------
    ax = axes[0]
    cols = plt.get_cmap(S.CAT)(np.linspace(0.08, 0.72, len(TAUS)))
    for tau, c, dy in zip(TAUS, cols, (0.06, 0.17, 0.28)):
        a = matched_alpha(ETA, tau)
        ax.semilogx(LAM, gd_filter(LAM, ETA, tau), color=c, lw=2.6)
        ax.semilogx(LAM, ridge_filter(LAM, a), color=c, lw=2.0,
                    ls=(0, (4, 2.5)))
        ax.axvline(a, color=c, lw=1.0, ls=(0, (1.5, 2.5)), alpha=0.8)
        ax.annotate(r"$\tau = %d$" % tau, xy=(a * 1.30, dy),
                    color=c, fontsize=13, ha="left")
    ax.plot([], [], color=S.GREY, lw=2.6, label=r"stopping at $\tau$")
    ax.plot([], [], color=S.GREY, lw=2.0, ls=(0, (4, 2.5)),
            label=r"decay at $\alpha = 1/\eta\tau$")
    ax.set_xlabel(r"curvature $\lambda_i$")
    ax.set_ylabel("fraction of $w^\\star_i$ retained")
    ax.set_ylim(-0.03, 1.08)
    ax.legend(loc="upper left", fontsize=12.5)
    ax.set_title("(a) the same filter, twice")
    S.square(ax)

    # ---- (b) the surviving directions ---------------------------------
    ax = axes[1]
    taus = np.logspace(0.3, 4.2, 90)
    n_stop = np.array([gd_filter(SPEC, ETA, int(round(t))).sum()
                       for t in taus])
    alphas = matched_alpha(ETA, taus)
    n_decay = np.array([ridge_filter(SPEC, a).sum() for a in alphas])

    ax.semilogx(taus, n_stop, color=S.OUTC, lw=2.8,
                label=r"stopping at $\tau$")
    ax.semilogx(taus, n_decay, color=S.L2, lw=2.2, ls=(0, (4, 2.5)),
                label=r"decay at $\alpha = 1/\eta\tau$")
    ax.axhline(len(SPEC), color=S.GREY, lw=1.1, ls=(0, (2, 3)))
    ax.annotate(r"$W = %d$ parameters" % len(SPEC),
                xy=(1.6e1, len(SPEC) * 0.90), color=S.GREY, fontsize=12.5,
                va="top")
    gap = np.max(np.abs(n_stop - n_decay)) / len(SPEC)
    ax.annotate("largest disagreement:\n%.1f%% of $W$" % (100 * gap),
                xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
                fontsize=12.5, color=S.GREY)
    ax.set_xlabel(r"run length $\tau$")
    ax.set_ylabel("directions still alive")
    ax.set_ylim(0, 1.18 * len(SPEC))
    ax.legend(loc="upper left", fontsize=12.5)
    ax.set_title("(b) a run length is a penalty strength")
    S.square(ax)

    fig.tight_layout(w_pad=1.4)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

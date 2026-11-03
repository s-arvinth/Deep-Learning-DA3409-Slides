"""condition_from_data — two ways the data alone can make the problem badly conditioned

Used in: Class 15 (B), slide(s): "Two ways the data alone can wreck the conditioning"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python condition_from_data.py            -> output/condition_from_data.pdf
         python condition_from_data.py --out DIR  -> DIR/condition_from_data.pdf
Ported from: Class15_Normalization/B_Recreated_Figures/src/figures/fig02_condition_from_data.py

(a) a difference in scale between two features: the condition number of
the Hessian grows as the square of the ratio, and normalization removes
it exactly.

(b) correlation between two features of equal scale: the condition
number grows without bound as the correlation approaches one, and
normalization does NOT remove it. Per-feature scaling fixes the
diagonal; it cannot fix the off-diagonal.
"""
import numpy as np
import matplotlib.pyplot as plt
import common as S
from models import (two_feature_data, standardize, linear_hessian,
                           condition_number)

NAME = "condition_from_data"


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2))

    ax = axes[0]
    ratios = np.geomspace(1.0, 300.0, 25)
    raw, nrm = [], []
    for r in ratios:
        X = two_feature_data(n=4000, scale=(1.0, r), seed=5)
        raw.append(condition_number(linear_hessian(X)))
        nrm.append(condition_number(linear_hessian(standardize(X)[0])))
    ax.loglog(ratios, raw, "o-", color=S.OUTC, lw=2.2, ms=6, mfc="white",
              mew=1.6, label="raw features")
    ax.loglog(ratios, nrm, "o-", color=S.L2, lw=2.2, ms=6, mfc="white",
              mew=1.6, label="after normalization")
    ax.loglog(ratios, ratios ** 2, "--", color=S.GREY, lw=1.6,
              label=r"$(\sigma_2/\sigma_1)^2$")
    ax.set_xlabel(r"ratio of feature scales $\sigma_2/\sigma_1$")
    ax.set_ylabel(r"condition number $\kappa$")
    ax.set_ylim(0.5, 1e6)
    ax.legend(loc="upper left", fontsize=12.5)
    ax.set_title("(a) unequal scale: normalization removes it")
    S.square(ax)

    ax = axes[1]
    cs = np.array([0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995])
    raw, nrm = [], []
    for c in cs:
        X = two_feature_data(n=8000, scale=(1.0, 1.0), corr=c, seed=7)
        raw.append(condition_number(linear_hessian(X)))
        nrm.append(condition_number(linear_hessian(standardize(X)[0])))
    ax.semilogy(cs, raw, "o-", color=S.OUTC, lw=2.2, ms=6, mfc="white",
                mew=1.6, label="raw features")
    ax.semilogy(cs, nrm, "s--", color=S.L2, lw=2.0, ms=5.5, mfc="white",
                mew=1.6, label="after normalization")
    ax.set_xlabel("correlation between the two features")
    ax.set_ylabel(r"condition number $\kappa$")
    ax.set_ylim(0.5, 1e4)
    ax.annotate("the two curves coincide", xy=(0.30, 3.4),
                fontsize=13.5, color=S.GREY)
    ax.legend(loc="upper left", fontsize=12.5)
    ax.set_title("(b) correlation: normalization does not")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

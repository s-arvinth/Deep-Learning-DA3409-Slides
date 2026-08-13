"""Diminishing returns from a larger batch.

Bishop & Bishop (2024), Section 7.2.4: the error in estimating a mean
from B samples is sigma / sqrt(B), so raising the batch size by a factor
of one hundred buys a factor of only ten in the accuracy of the
gradient.  That single square root is the entire argument for
mini-batches.

Panel (a) is the sampling distribution of one component of the estimate
at three batch sizes.  Panel (b) is the standard error against B on log
axes, with the hundred-to-ten trade annotated.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import batch_gradient_samples

NAME = "batch_variance"


def build():
    S.use()
    rng = np.random.default_rng(11)
    N = 4096
    # per-example gradients: same mean, heavy per-example spread
    per_point = rng.normal(loc=1.0, scale=2.6, size=(N, 1))
    truth = per_point.mean()

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2))

    ax = axes[0]
    cols = plt.get_cmap(S.CAT)(np.linspace(0.08, 0.74, 3))
    for B, col in zip((1, 16, 256), cols):
        s = batch_gradient_samples(per_point, B, 6000, seed=B)[:, 0]
        ax.hist(s, bins=70, density=True, histtype="stepfilled",
                color=col, alpha=0.42, zorder=2)
        ax.hist(s, bins=70, density=True, histtype="step",
                color=col, lw=1.9, zorder=3, label=r"$B = %d$" % B)
    ax.axvline(truth, color=S.OUTC, lw=1.8, ls=(0, (5, 3)), zorder=4)
    ax.annotate("full-batch\ngradient", xy=(truth, 1.05), fontsize=13.5,
                color=S.OUTC, ha="center", va="bottom")
    ax.set_xlim(-5.5, 7.5)
    ax.set_ylim(0, 1.45)
    ax.set_xlabel("one component of the estimate")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=13)
    ax.set_title("(a) the estimate is unbiased, not exact")
    S.square(ax)

    ax = axes[1]
    Bs = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    se = np.array([batch_gradient_samples(per_point, int(B), 1500,
                                          seed=int(B))[:, 0].std()
                   for B in Bs])
    ax.loglog(Bs, se, "o-", color=S.L1, lw=2.0, ms=7, mfc="white",
              mew=1.8, label="measured")
    ax.loglog(Bs, per_point.std() / np.sqrt(Bs), "--", color=S.GREY,
              lw=1.8, label=r"$\sigma/\sqrt{B}$")
    b0, b1 = 4.0, 400.0
    s0, s1 = per_point.std() / np.sqrt(b0), per_point.std() / np.sqrt(b1)
    ax.plot([b0, b1], [s0, s0], color=S.OUTC, lw=1.6)
    ax.plot([b1, b1], [s0, s1], color=S.OUTC, lw=1.6)
    ax.plot([b0], [s0], "o", ms=6, color=S.OUTC)
    ax.plot([b1], [s1], "o", ms=6, color=S.OUTC)
    ax.annotate(r"$\times 100$ the batch", xy=(40, s0 * 1.22),
                fontsize=14.5, color=S.OUTC, ha="center")
    ax.annotate(r"$\div\,10$ the error", xy=(b1 * 1.35, s0 * 0.34),
                fontsize=14.5, color=S.OUTC, ha="left", va="center")
    ax.set_xlabel("batch size $B$")
    ax.set_ylabel("standard error of the estimate")
    ax.set_xlim(0.7, 3000)
    ax.legend(loc="lower left", fontsize=13)
    ax.set_title("(b) the square root is the whole argument")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

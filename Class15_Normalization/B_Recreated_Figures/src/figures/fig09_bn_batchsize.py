"""Batch normalization estimates its statistics from the batch.

The mean and variance it subtracts are sample estimates from B points,
so their own standard error falls only as one over the root of B. Below
a few dozen examples the correction being applied is mostly noise.

(a) the error in the estimated mean and standard deviation against the
batch size. (b) the gap between the statistics used during training and
the ones used at inference time, which is what that noise turns into.
Layer normalization uses no batch statistics at all and is flat.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S

NAME = "bn_batchsize"

BS = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512])
DRAWS = 4000


def build():
    S.use()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))

    ax = axes[0]
    em, es = [], []
    for B in BS:
        x = rng.standard_normal((DRAWS, int(B)))
        em.append(x.mean(axis=1).std())
        es.append(x.std(axis=1).std())
    ax.loglog(BS, em, "o-", color=S.OUTC, lw=2.2, ms=6.5, mfc="white",
              mew=1.6, label=r"error in $\mu$")
    ax.loglog(BS, es, "s-", color=S.L2, lw=2.2, ms=6, mfc="white",
              mew=1.6, label=r"error in $\sigma$")
    ax.loglog(BS, 1.0 / np.sqrt(BS), "--", color=S.GREY, lw=1.7,
              label=r"$1/\sqrt{B}$")
    ax.set_xlabel("batch size $B$")
    ax.set_ylabel("standard error of the estimate")
    ax.legend(loc="upper right", fontsize=13)
    ax.set_title("(a) the statistics are themselves estimates")
    S.square(ax)

    ax = axes[1]
    bn = np.array(em) + np.array(es)
    ax.semilogx(BS, bn, "o-", color=S.OUTC, lw=2.2, ms=6.5, mfc="white",
                mew=1.6, label="batch normalization")
    ax.semilogx(BS, np.zeros_like(BS, dtype=float), "s-", color=S.L2,
                lw=2.2, ms=6, mfc="white", mew=1.6,
                label="layer normalization")
    ax.axvspan(1.5, 16, color=S.OUTC, alpha=0.10)
    ax.annotate("where batch norm\nstops being reliable",
                xy=(5.5, 0.62), fontsize=13, ha="center", color=S.OUTC)
    ax.set_xlabel("batch size $B$")
    ax.set_ylabel("train-to-inference mismatch")
    ax.set_ylim(-0.06, 1.15)
    ax.legend(loc="upper right", fontsize=13)
    ax.set_title("(b) layer norm does not depend on $B$")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()

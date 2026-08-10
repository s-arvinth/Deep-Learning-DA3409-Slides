#!/usr/bin/env python3
"""Universal approximation with a rate: the error of the best M-unit
one-hidden-layer fit, against the 1/sqrt(M) reference from Barron (1993)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.data import smooth_target
from common.models import best_pwl

NAME = "uat_rate"


def build():
    style.use()
    x = np.linspace(0, 1, 4001)
    f = smooth_target(x)

    fig, ax = plt.subplots(2, 2, figsize=(9.6, 8.6))
    for a_, M, c, tag in zip(ax.ravel()[:3], [4, 16, 64],
                             [L1, L2, OUTC], "abc"):
        yh = best_pwl(x, f, M + 1)
        err = np.abs(f - yh).max()
        a_.plot(x, f, color=GREY, lw=2.2, ls="--", label=r"target $f$")
        a_.plot(x, yh, color=c, lw=2.2, label=rf"$M={M}$ units")
        a_.legend(loc="upper right", fontsize=11)
        a_.set(xlabel=r"$x$", ylabel=r"$y$")
        a_.set_title(rf"({tag})  $M={M}$:  "
                     rf"$\sup|f-\hat y|={err:.4f}$", fontsize=12)
        style.square(a_)

    Ms = np.unique(np.round(np.logspace(0.3, 2.4, 20)).astype(int))
    errs = np.array([np.abs(f - best_pwl(x, f, int(M) + 1)).max() for M in Ms])
    a_ = ax[1, 1]
    a_.loglog(Ms, errs, color=OUTC, lw=2.4, marker="o", ms=6, mec="white",
              mew=0.8, label=r"measured $\sup|f-\hat y|$")
    a_.loglog(Ms, errs[0] * (Ms / Ms[0]) ** -0.5, color=L1, lw=2.0, ls="--",
              label=r"Barron:  $M^{-1/2}$")
    a_.loglog(Ms, errs[0] * (Ms / Ms[0]) ** -2.0, color=L2, lw=2.0, ls=":",
              label=r"smooth, $D=1$:  $M^{-2}$")
    a_.legend(loc="lower left", fontsize=10.5)
    a_.set(xlabel=r"hidden units $M$", ylabel=r"maximum error")
    a_.set_title(r"(d)  the rate, measured", fontsize=12.5)
    style.square(a_)

    fig.tight_layout(w_pad=2.4, h_pad=2.6)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

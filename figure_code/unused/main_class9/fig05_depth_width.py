#!/usr/bin/env python3
"""Depth versus width on the sawtooth: a deep network represents it
exactly with O(L) parameters, while a shallow one cannot get close
until its width reaches 2^L - 1."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import tooth_iter, best_pwl, n_params

NAME = "depth_width"
LD = 3          # target is T composed LD times: 2^LD teeth


def build():
    style.use()
    x = np.linspace(0, 1, 40001)
    f = tooth_iter(x, LD)

    fig, ax = plt.subplots(2, 2, figsize=(9.6, 8.6))
    for a_, M, c, tag in zip(ax.ravel()[:3], [4, 8, 2 ** LD - 1],
                             [L1, L2, OUTC], "abc"):
        yh = best_pwl(x, f, M + 1)
        err = np.abs(f - yh).max()
        a_.plot(x, f, color=GREY, lw=1.8, ls="--", label=r"target")
        a_.plot(x, yh, color=c, lw=2.2, label=rf"$M={M}$")
        a_.legend(loc="upper right", fontsize=10.5, framealpha=.92,
                  facecolor="white", edgecolor="none")
        a_.set(xlabel=r"$x$", ylabel=r"$y$", ylim=(-0.08, 1.16))
        a_.set_title(rf"({tag})  shallow, $M={M}$:  "
                     rf"$\sup|f-\hat y|={err:.3f}$", fontsize=11.5)
        style.square(a_)

    Ms = np.arange(2, 9)
    errs = np.array([np.abs(f - best_pwl(x, f, int(M) + 1)).max() for M in Ms])
    a_ = ax[1, 1]
    a_.semilogy(Ms, np.maximum(errs, 1e-7), color=L1, lw=2.4,
                marker="o", ms=7, mec="white", mew=0.8,
                label=r"shallow, width $M$")
    a_.axvline(2 ** LD - 1, color=OUTC, lw=2.2, ls="--")
    a_.annotate(rf"$M=2^{{{LD}}}-1={2**LD-1}$: the first" "\n"
                r"width with enough pieces",
                xy=(2.15, 2e-5), color=OUTC, fontsize=11)
    a_.legend(loc="lower left", fontsize=11)
    a_.set(xlabel=r"shallow width $M$", ylabel=r"maximum error",
           ylim=(3e-8, 3), xlim=(1.6, 8.4))
    style.integer_ticks(a_, "x", nbins=7)
    a_.set_title(rf"(d)  the deep net uses ${n_params(2, LD)}$ parameters "
                 rf"and is exact", fontsize=11)
    style.square(a_)

    fig.tight_layout(w_pad=2.4, h_pad=2.6)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

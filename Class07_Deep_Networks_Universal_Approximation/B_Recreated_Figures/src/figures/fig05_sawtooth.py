#!/usr/bin/env python3
"""The sawtooth: the witness for the depth-versus-width separation.

T(x) = 2h(x) - 4h(x - 1/2) is a two-unit ReLU network folding [0,1] in
half.  Composing it L times gives 2^L linear pieces from O(L) units,
while a shallow ReLU network needs at least 2^L units to match it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import tooth_iter

NAME = "sawtooth"


def build():
    style.use()
    x = np.linspace(0, 1, 20001)

    fig, ax = plt.subplots(2, 2, figsize=(9.6, 8.6))
    for a_, Lk, c, tag in zip(ax.ravel()[:3], [1, 2, 4],
                              [L1, L2, OUTC], "abc"):
        a_.plot(x, tooth_iter(x, Lk), color=c, lw=2.2)
        a_.set(xlabel=r"$x$", ylabel=rf"$T^{{\circ {Lk}}}(x)$",
               ylim=(-0.05, 1.05))
        a_.set_title(rf"({tag})  $L={Lk}$:  $2^{{{Lk}}}={2**Lk}$ pieces "
                     rf"from ${2*Lk}$ units", fontsize=12.5)
        style.square(a_)

    Ls = np.arange(1, 17)
    a_ = ax[1, 1]
    a_.semilogy(Ls, 2.0 ** Ls, color=OUTC, lw=2.4, marker="o", ms=6,
                mec="white", mew=0.8,
                label=r"depth $L$, width 2:  $2^{L}$")
    a_.semilogy(Ls, 2.0 * Ls + 1, color=L1, lw=2.4, marker="s", ms=6,
                mec="white", mew=0.8, ls="--",
                label=r"one layer, same $2L$ units:  $2L+1$")
    a_.legend(loc="upper left", fontsize=11.5)
    a_.set(xlabel=r"hidden layers $L$  (2 units each)",
           ylabel="linear pieces")
    a_.set_title(r"(d)  same unit budget, spent two ways", fontsize=12.5)
    style.square(a_)

    fig.tight_layout(w_pad=2.4, h_pad=2.6)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()

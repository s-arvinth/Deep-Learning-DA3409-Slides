#!/usr/bin/env python3
"""The ReLU activation h(a) = max(0, a)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import relu

NAME = "relu"


def build():
    style.use()
    a = np.linspace(-2, 2, 600)
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot(a, a, color=GREY, lw=1.3, ls="--", label=r"$a$  (identity)")
    ax.plot(a, relu(a), color=OUTC, lw=2.8, label=r"$h(a)=\max(0,a)$")
    ax.axhline(0, color=GREY, lw=0.6, ls=":")
    ax.axvline(0, color=GREY, lw=0.6, ls=":")
    ax.set_ylim(-1.05, 2.1)
    ax.annotate("inactive", xy=(-1.5, 0.12), color=GREY, fontsize=11)
    ax.annotate("active", xy=(0.75, 0.45), color=OUTC, fontsize=11)
    ax.legend(loc="upper left", fontsize=11)
    ax.set(xlabel=r"pre-activation  $a$", ylabel=r"activation  $h(a)$",
           title=r"the rectified linear unit")
    ax.title.set_fontsize(12.5)
    style.square(ax)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()
